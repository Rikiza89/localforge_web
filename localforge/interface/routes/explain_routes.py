"""
説明ルート — /api/explain/* エンドポイントの定義。
インデックス構築・レポート生成・Q&A（SSEストリーミング）を提供する。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import List

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from localforge.application.analysis_service import AnalysisService
from localforge.application.explanation_service import ExplanationService
from localforge.application.project_service import ProjectService
from localforge.domain.exceptions import LocalForgeError
from localforge.domain.models import Message

logger = logging.getLogger(__name__)

bp = Blueprint("explain", __name__, url_prefix="/api/explain")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream",
}


def _get_project_svc() -> ProjectService:
    return current_app.config["project_service"]


def _get_analysis_svc() -> AnalysisService:
    return current_app.config["analysis_service"]


def _get_explanation_svc() -> ExplanationService:
    return current_app.config["explanation_service"]


_HEARTBEAT_INTERVAL = 15  # 秒

_HB = {"heartbeat": True}
_HB_LINE = f"data: {json.dumps(_HB, ensure_ascii=False)}\n\n"


def _sse_response(generator):
    """
    SSEレスポンスを生成する。
    ハートビートはバックグラウンドスレッドからキューに投入するため、
    LLM呼び出しでジェネレーターがブロックされていても15秒ごとに送出される。
    """
    def wrapped():
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def _produce():
            try:
                for payload in generator:
                    if stop.is_set():
                        break
                    q.put(payload)
            except Exception as exc:
                q.put({"error": str(exc)})
            finally:
                q.put(None)  # 終了センチネル

        def _heartbeat():
            while not stop.wait(_HEARTBEAT_INTERVAL):
                q.put(_HB)

        threading.Thread(target=_produce, daemon=True).start()
        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            while True:
                payload = q.get()
                if payload is None:
                    break
                if "token" in payload:
                    tok = payload["token"]
                    if tok.startswith("\x01"):
                        # 思考トークン: Ollamaパネル専用（メイン表示には送らない）
                        thinking_text = tok[1:]
                        yield f"data: {json.dumps({'raw_token': '<think>' + thinking_text + '</think>'}, ensure_ascii=False)}\n\n"
                        continue  # メイン token イベントを送出しない
                    else:
                        yield f"data: {json.dumps({'raw_token': tok}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            stop.set()

    return Response(
        stream_with_context(wrapped()),
        mimetype="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _error_response(exc: Exception, status: int = 500):
    return jsonify({"error": type(exc).__name__, "message": str(exc)}), status


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

    gen = explanation_svc.stream_report(
        root=root,
        model=model,
        selected_section_indices=selected_indices,
        resume_from=resume_from,
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
