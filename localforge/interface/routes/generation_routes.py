"""
生成ルート — /api/generate/* エンドポイントの定義。
プラン生成・承認・ファイル生成（SSEストリーミング）を提供する。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from localforge.application.analysis_service import AnalysisService
from localforge.application.generation_service import (
    GenerationService,
    request_cancel,
    reset_cancel,
)
from localforge.application.project_service import ProjectService
from localforge.domain.exceptions import LocalForgeError, PlanParseError
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter

logger = logging.getLogger(__name__)

bp = Blueprint("generation", __name__, url_prefix="/api/generate")

# SSEヘッダー
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream",
}


def _get_generation_svc() -> GenerationService:
    return current_app.config["generation_service"]


def _get_project_svc() -> ProjectService:
    return current_app.config["project_service"]


def _get_git() -> GitAdapter:
    return current_app.config["git"]


def _get_analysis_svc() -> AnalysisService:
    return current_app.config["analysis_service"]


def _get_index_adapter() -> IndexAdapter:
    return current_app.config["index_adapter"]


_HEARTBEAT_INTERVAL = 15  # 秒

_HB = {"heartbeat": True}


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
                q.put(None)

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
                        yield f"data: {json.dumps({'raw_token': '<think>' + thinking_text + '</think>'})}\n\n"
                        continue
                    else:
                        yield f"data: {json.dumps({'raw_token': tok})}\n\n"
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            stop.set()

    return Response(
        stream_with_context(wrapped()),
        mimetype="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _error_response(exc: Exception, status: int = 500):
    return jsonify({"error": type(exc).__name__, "message": str(exc)}), status


@bp.route("/plan", methods=["POST"])
def stream_plan():
    """
    ユーザープロンプトからプロジェクト生成プランをSSEストリーミングする。

    Request JSON:
        prompt (str): ユーザーの自然言語プロンプト

    SSE Events:
        token, done, error
    """
    project_svc = _get_project_svc()
    generation_svc = _get_generation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    user_prompt = data.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "NoPrompt", "message": "プロンプトが指定されていません"}), 400

    model = project.config.model
    if not model:
        return jsonify({"error": "NoModel", "message": "モデルが選択されていません。UIでモデルを選択してください"}), 400

    root = project.root

    # ファイルツリーのテキスト表現を構築
    file_tree_text = _build_tree_text(root)
    context_md = project_svc.get_context_md(root)
    git_log_entries = _get_git().get_log(root, max_entries=5)
    git_log = "\n".join(
        f"- {e['hash']} {e['message']}" for e in git_log_entries
    )

    # E: RAGで既存ファイルの関連サマリーを取得してプランプロンプトに注入する
    file_summaries = []
    try:
        index_path = root / ".localforge" / "index.jsonl"
        chunks = _get_index_adapter().load_chunks(index_path)
        if chunks:
            top_chunks = _get_analysis_svc().get_top_chunks_semantic(
                chunks, user_prompt, top_n=15
            )
            file_summaries = [(c.path, c.summary) for c in top_chunks if c.summary]
    except Exception as exc:
        logger.warning("RAGファイルサマリー取得エラー: %s", exc)

    gen = generation_svc.stream_plan(
        root=root,
        model=model,
        user_prompt=user_prompt,
        folder_name=root.name,
        file_tree_text=file_tree_text,
        context_md=context_md,
        git_log=git_log,
        file_summaries=file_summaries,
    )
    return _sse_response(gen)


@bp.route("/approve", methods=["POST"])
def approve_plan():
    """
    生成されたプランを承認して.localforge/plan.jsonに保存する。

    Request JSON:
        plan_json (str): プランのJSON文字列

    Response JSON:
        plan: 承認されたプランの情報
    """
    project_svc = _get_project_svc()
    generation_svc = _get_generation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    plan_json = data.get("plan_json", "").strip()
    if not plan_json:
        return jsonify({"error": "NoPlan", "message": "プランが指定されていません"}), 400

    try:
        plan = generation_svc.parse_plan(plan_json)
    except PlanParseError as exc:
        return _error_response(exc, 400)
    except LocalForgeError as exc:
        return _error_response(exc)

    plan.approved = True
    project_svc.save_generation_plan(project.root, plan)

    return jsonify({
        "plan": {
            "project_name": plan.project_name,
            "description": plan.description,
            "file_count": len(plan.files),
            "files": [{"path": f.path, "description": f.description} for f in plan.files],
        }
    })


@bp.route("/start", methods=["GET"])
def stream_generation():
    """
    承認済みプランに基づいてすべてのファイルをSSEストリーミング生成する。

    SSE Events:
        progress, token, file_written, done, error
    """
    project_svc = _get_project_svc()
    generation_svc = _get_generation_svc()
    project = project_svc.current_project
    if not project:
        def err_gen():
            yield {"error": "プロジェクトが開かれていません"}
        return _sse_response(err_gen())

    plan = project_svc.load_generation_plan(project.root)
    if not plan:
        def err_gen():
            yield {"error": "承認済みプランが見つかりません"}
        return _sse_response(err_gen())

    model = project.config.model
    if not model:
        def err_gen():
            yield {"error": "モデルが選択されていません。UIでモデルを選択してください"}
        return _sse_response(err_gen())

    root = project.root
    context_md = project_svc.get_context_md(root)

    gen = generation_svc.stream_all_files(
        root=root,
        plan=plan,
        model=model,
        context_md=context_md,
    )
    return _sse_response(gen)


@bp.route("/cancel", methods=["POST"])
def cancel_generation():
    """
    現在実行中の生成処理をキャンセルする。

    Response JSON:
        cancelled: true
    """
    request_cancel()
    return jsonify({"cancelled": True})


@bp.route("/regenerate", methods=["POST"])
def regenerate_file():
    """
    単一ファイルを再生成してSSEストリーミングする。

    Request JSON:
        file_path (str): 再生成するファイルの相対パス

    SSE Events:
        token, file_written, done, error
    """
    project_svc = _get_project_svc()
    generation_svc = _get_generation_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    file_path = data.get("file_path", "").strip()
    if not file_path:
        return jsonify({"error": "NoFilePath", "message": "ファイルパスが指定されていません"}), 400

    plan = project_svc.load_generation_plan(project.root)
    if not plan:
        return jsonify({"error": "NoPlan", "message": "承認済みプランが見つかりません"}), 400

    model = project.config.model
    if not model:
        return jsonify({"error": "NoModel", "message": "モデルが選択されていません。UIでモデルを選択してください"}), 400

    root = project.root
    context_md = project_svc.get_context_md(root)

    gen = generation_svc.stream_regenerate_file(
        root=root,
        plan=plan,
        model=model,
        context_md=context_md,
        file_path=file_path,
    )
    return _sse_response(gen)


def _build_tree_text(root: Path, path: Path | None = None, prefix: str = "") -> str:
    """
    ディレクトリツリーをテキストで表現する補助関数。

    Args:
        root: ルートディレクトリ
        path: 現在のパス（省略時はroot）
        prefix: インデントプレフィックス

    Returns:
        ツリーテキスト
    """
    skip_dirs = {".git", ".localforge", "__pycache__", "node_modules", ".venv", "venv"}
    if path is None:
        path = root

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return ""

    lines = []
    filtered = [e for e in entries if not (e.is_dir() and e.name in skip_dirs)]

    for i, entry in enumerate(filtered):
        is_last = i == len(filtered) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            subtree = _build_tree_text(root, entry, prefix + extension)
            if subtree:
                lines.append(subtree)

    return "\n".join(lines)
