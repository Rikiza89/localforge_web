"""
説明ルート — /api/explain/* エンドポイントの定義。
インデックス構築・レポート生成・Q&A（SSEストリーミング）を提供する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from flask import Blueprint, current_app, jsonify, request

from localforge.application.explanation_service import ExplanationService
from localforge.domain.exceptions import LocalForgeError
from localforge.domain.models import Message

logger = logging.getLogger(__name__)

bp = Blueprint("explain", __name__, url_prefix="/api/explain")

from localforge.interface.routes._sse_helpers import (  # noqa: E402
    _SSE_HEADERS, _HB, _HEARTBEAT_INTERVAL,
    _sse_response, _error_response,
    _get_project_svc, _get_analysis_svc,
)


def _get_explanation_svc() -> ExplanationService:
    return current_app.config["explanation_service"]


@bp.route("/sections", methods=["GET"])
def get_report_sections():
    """
    Return the ordered list of report section names.
    Used by the frontend to build the section-selector checkboxes dynamically.
    """
    from localforge.application.explanation_service import REPORT_SECTIONS
    return jsonify({"sections": REPORT_SECTIONS})


@bp.route("/search", methods=["GET"])
def search_project():
    """
    プロジェクト全体のセマンティック検索（ChromaDB / BM25フォールバック）。

    Query params:
        q: 検索クエリ
        top_n: 返す件数（デフォルト10、最大30）

    Response JSON:
        results: [{path, summary, language}]
    """
    project_svc = _get_project_svc()
    analysis_svc = _get_analysis_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})

    try:
        top_n = max(1, min(int(request.args.get("top_n", "10")), 30))
    except ValueError:
        top_n = 10

    project_index = analysis_svc.load_project_index(project.root)
    if not project_index:
        return jsonify({"error": "NoIndex", "message": "先にインデックスを構築してください"}), 404

    results = analysis_svc.get_top_chunks_semantic(
        project_index.file_chunks, query, top_n=top_n
    )
    return jsonify({
        "results": [
            {
                "path": c.path,
                "summary": (c.summary or "")[:300],
                "language": c.language or "",
            }
            for c in results
        ]
    })


@bp.route("/index", methods=["GET"])
def stream_index():
    """
    コードベースのインデックスを構築してSSEで進捗をストリーミングする。
    増分インデックス: 変更されたファイルのみ再処理される。

    SSE Events:
        progress, done, error
    """
    project_svc = _get_project_svc()
    analysis_svc = _get_analysis_svc()
    project = project_svc.current_project
    if not project:
        def err_gen():
            yield {"error": "プロジェクトが開かれていません"}
        return _sse_response(err_gen())

    model = project.config.model
    if not model:
        def err_gen():
            yield {"error": "モデルが選択されていません。UIでモデルを選択してください"}
        return _sse_response(err_gen())

    root = project.root

    try:
        gen = analysis_svc.build_index(root=root, model=model)
        return _sse_response(gen)
    except LocalForgeError as exc:
        def err_gen():
            yield {"error": str(exc)}
        return _sse_response(err_gen())


@bp.route("/report", methods=["GET"])
def stream_report():
    """
    レポートをSSEストリーミングする。
    クエリパラメータ:
      - sections: カンマ区切りのセクションインデックス (例: 0,1,4,7)
      - resume_from: このインデックス以降を生成 (例: 6)
      - model: 使用モデルの上書き（省略時はプロジェクト設定モデル）

    SSE Events:
        section, token, progress, done, error
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        def err_gen():
            yield {"error": "プロジェクトが開かれていません"}
        return _sse_response(err_gen())

    model = request.args.get("model", "").strip() or project.config.model
    if not model:
        def err_gen():
            yield {"error": "モデルが選択されていません。UIでモデルを選択してください"}
        return _sse_response(err_gen())

    root = project.root

    # セクション選択フィルタ
    sections_param = request.args.get("sections", "").strip()
    selected_indices: List[int] | None = None
    if sections_param:
        try:
            selected_indices = [int(x) for x in sections_param.split(",") if x.strip().isdigit()]
        except ValueError:
            selected_indices = None

    # resume_from
    resume_from = 0
    try:
        resume_from = max(0, int(request.args.get("resume_from", "0")))
    except ValueError:
        resume_from = 0

    language = request.args.get("lang", "ja").strip().lower()
    if language not in ("ja", "en"):
        language = "ja"

    gen = explanation_svc.stream_report(
        root=root,
        model=model,
        selected_section_indices=selected_indices,
        resume_from=resume_from,
        language=language,
    )
    return _sse_response(gen)


@bp.route("/ask", methods=["POST"])
def ask_question():
    """
    プロジェクトに関する質問への回答をSSEストリーミングする。

    Request JSON:
        question (str): ユーザーの質問
        history (list, optional): 会話履歴 [{role, content}, ...]

    SSE Events:
        token, done, error
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "NoQuestion", "message": "質問が指定されていません"}), 400

    raw_history = data.get("history", [])
    history: List[Message] = []
    for item in raw_history[-10:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            history.append(Message(role=role, content=content))

    mode = data.get("mode", "ultra")
    if mode not in ("precise", "fast", "ultra"):
        mode = "ultra"
    model = project.config.model
    root = project.root

    # ワークスペースプロジェクトと ピン留めコンテキストを解決して渡す
    workspace_roots = project_svc.get_workspace_roots(root)
    pinned_paths = project_svc.get_pinned_context(root)

    gen = explanation_svc.stream_answer(
        root=root,
        model=model,
        question=question,
        history=history,
        workspace_roots=workspace_roots or None,
        pinned_paths=pinned_paths or None,
        mode=mode,
    )
    return _sse_response(gen)


@bp.route("/qa-save", methods=["POST"])
def save_qa_entry():
    """
    Q&Aのやり取りを .localforge/qa_history.md に追記する。

    Request JSON:
        question (str): ユーザーの質問
        answer (str): アシスタントの回答

    Response JSON:
        ok: true
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    if not question or not answer:
        return jsonify({"error": "InvalidData", "message": "questionとanswerは必須です"}), 400

    try:
        explanation_svc.append_qa_entry(project.root, question, answer)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Q&A保存エラー: %s", exc)
        return jsonify({"error": "SaveError", "message": str(exc)}), 500


@bp.route("/saved-report", methods=["GET"])
def get_saved_report():
    """
    .localforge/report.md の内容を返す。

    Response JSON:
        content: Markdownテキスト（存在しない場合はnull）
        partial: 部分的なレポートかどうか
        sections_done: 完了済みセクション数（部分的な場合）
        sections_total: 総セクション数（部分的な場合）
    """
    import re as _re
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    report_path = project.root / ".localforge" / "report.md"
    if not report_path.exists():
        return jsonify({"content": None, "partial": False})

    try:
        content = report_path.read_text(encoding="utf-8")
        m = _re.search(r"<!-- localforge:partial:(\d+)/(\d+) -->", content)
        partial = m is not None
        return jsonify({
            "content": content,
            "partial": partial,
            "sections_done": int(m.group(1)) if m else None,
            "sections_total": int(m.group(2)) if m else None,
        })
    except OSError as exc:
        return jsonify({"error": "ReadError", "message": str(exc)}), 500


@bp.route("/report-history", methods=["GET"])
def get_report_history():
    """
    レポート履歴メタデータ一覧を返す（新しい順）。

    Response JSON: [{id, filename, created_at, partial, sections_done, sections_total, model}]
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    history = explanation_svc.get_report_history(project.root)
    return jsonify({"history": history})


@bp.route("/report-history/<report_id>", methods=["GET"])
def get_historical_report(report_id: str):
    """
    指定IDの履歴レポート内容を返す。

    Response JSON:
        content: Markdownテキスト
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    content = explanation_svc.get_historical_report(project.root, report_id)
    if content is None:
        return jsonify({"error": "NotFound", "message": "指定されたレポートが見つかりません"}), 404
    return jsonify({"content": content})


@bp.route("/report-history/<report_id>", methods=["DELETE"])
def delete_historical_report(report_id: str):
    """
    指定IDの履歴レポートを削除する。

    Response JSON:
        ok: true
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    ok = explanation_svc.delete_historical_report(project.root, report_id)
    if not ok:
        return jsonify({"error": "NotFound", "message": "指定されたレポートが見つかりません"}), 404
    return jsonify({"ok": True})


@bp.route("/qa-history", methods=["GET"])
def get_qa_history():
    """
    .localforge/qa_history.md からQ&A履歴を読み込んで返す。

    Response JSON:
        entries: [{timestamp, question, answer}, ...]  (最大50件、新しい順)
    """
    import re as _re
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    qa_path = project.root / ".localforge" / "qa_history.md"
    if not qa_path.exists():
        return jsonify({"entries": []})

    try:
        content = qa_path.read_text(encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": "ReadError", "message": str(exc)}), 500

    entries = []
    pattern = _re.compile(
        r"##\s+\[([^\]]+)\]\s*\n+\*\*Q:\*\*\s*(.*?)\s*\n+\*\*A:\*\*\s*(.*?)(?=\n+---|\Z)",
        _re.DOTALL,
    )
    for m in pattern.finditer(content):
        entries.append({
            "timestamp": m.group(1).strip(),
            "question": m.group(2).strip(),
            "answer": m.group(3).strip(),
        })

    entries = entries[-50:]
    entries.reverse()
    return jsonify({"entries": entries})


@bp.route("/migrate-vector", methods=["GET"])
def migrate_vector_index():
    """
    既存のJSONLインデックスからChromaDBベクトルインデックスへ移行する。
    インデックス済みプロジェクトを初めてRAGに移行する際に使用する。
    進捗をSSEストリーミングで返す。

    SSE Events:
        progress, done, error
    """
    project_svc = _get_project_svc()
    analysis_svc = _get_analysis_svc()
    project = project_svc.current_project
    if not project:
        def err_gen():
            yield {"error": "プロジェクトが開かれていません"}
        return _sse_response(err_gen())

    gen = analysis_svc.migrate_vector_index(root=project.root)
    return _sse_response(gen)


@bp.route("/summary", methods=["GET"])
def get_summary():
    """
    ProjectIndexのサマリーをJSON形式で返す。

    Response JSON:
        project_name, summary, total_files, indexed_files, created_at, updated_at
    """
    project_svc = _get_project_svc()
    explanation_svc = _get_explanation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    summary = explanation_svc.get_summary(project.root)
    if not summary:
        return jsonify({
            "error": "NoIndex",
            "message": "ProjectIndexが見つかりません。先にインデックスを構築してください。"
        }), 404

    vector = current_app.config.get("vector")
    rag_ready = vector.collection_exists(project.root) if vector else False
    summary["rag_ready"] = rag_ready
    return jsonify(summary)
