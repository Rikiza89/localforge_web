"""
生成ルート — /api/generate/* エンドポイントの定義。
プラン生成・承認・ファイル生成（SSEストリーミング）を提供する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from localforge.application.generation_service import (
    GenerationService,
    request_cancel,
    reset_cancel,
)
from localforge.domain.exceptions import LocalForgeError, PlanParseError
from localforge.domain.models import GenerationPlan, PlannedFile
from localforge.infrastructure.index_adapter import IndexAdapter

logger = logging.getLogger(__name__)

bp = Blueprint("generation", __name__, url_prefix="/api/generate")

from localforge.interface.routes._sse_helpers import (  # noqa: E402
    _SSE_HEADERS, _HB, _HEARTBEAT_INTERVAL,
    _sse_response, _error_response,
    _get_project_svc, _get_generation_svc, _get_analysis_svc, _get_git,
)


def _get_index_adapter() -> IndexAdapter:
    return current_app.config["index_adapter"]


def _after_done(gen, callback):
    """Yield all events from gen; call callback() once after done:True is seen."""
    done_seen = False
    for event in gen:
        yield event
        if event.get("done") and not done_seen:
            done_seen = True
    if done_seen:
        try:
            callback()
        except Exception as exc:
            logger.warning("post-generation callback error: %s", exc)


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
    file_tree_text = _get_analysis_svc().build_tree_text(root)
    context_md = project_svc.get_context_md(root)
    git_log_entries = _get_git().get_log(root, max_entries=5)
    git_log = "\n".join(
        f"- {e['hash']} {e['message']}" for e in git_log_entries
    )

    # ProjectIndex（プロジェクト全体概要）を取得してプランプロンプトに注入する
    project_index_json = None
    file_summaries = []
    try:
        analysis_svc = _get_analysis_svc()
        pi_path = root / ".localforge" / "project_index.json"
        index_adapter = _get_index_adapter()
        project_index = index_adapter.load_index(pi_path)
        if project_index:
            import json as _json
            index_dict = project_index.model_dump(
                include={"project_name", "summary", "total_files", "indexed_files"}
            )
            index_dict["files"] = [c.path for c in project_index.file_chunks]
            project_index_json = _json.dumps(index_dict, ensure_ascii=False)
            # RAG: クエリに関連するファイルサマリーを追加
            top_chunks = analysis_svc.get_top_chunks_semantic(
                project_index.file_chunks, user_prompt, top_n=15
            )
            file_summaries = [(c.path, c.summary) for c in top_chunks if c.summary]
    except Exception as exc:
        logger.warning("ProjectIndex/RAGファイルサマリー取得エラー: %s", exc)

    # ── ピン留めコンテキスト ──
    pinned_contents: list = []
    try:
        pinned_paths = project_svc.get_pinned_context(root)
        if pinned_paths:
            pi_path = root / ".localforge" / "project_index.json"
            pi = _get_index_adapter().load_index(pi_path)
            if pi:
                pin_chunks, _ = analysis_svc.resolve_pinned_chunks(
                    root, pinned_paths, pi.file_chunks, max_total=15
                )
                _max_pin = 4000
                for pc in pin_chunks:
                    fp = root / pc.path
                    if fp.exists():
                        try:
                            pinned_contents.append((pc.path, fp.read_text(encoding="utf-8", errors="replace")[:_max_pin]))
                        except OSError:
                            pass
    except Exception as exc:
        logger.warning("プランピン留めコンテキスト解決エラー: %s", exc)

    # ── ワークスペースプロジェクトのサマリー ──
    workspace_summaries: list = []
    try:
        ws_roots = project_svc.get_workspace_roots(root)
        for ws_root, ws_name in ws_roots[:3]:
            ws_idx = _get_index_adapter().load_index(ws_root / ".localforge" / "project_index.json")
            if ws_idx:
                workspace_summaries.append((ws_name, ws_idx.summary[:300]))
    except Exception as exc:
        logger.warning("ワークスペースサマリー取得エラー: %s", exc)

    # Optional file-count constraints from the UI
    def _parse_int_param(name: str) -> "int | None":
        val = data.get(name, None)
        if val is None:
            return None
        try:
            n = int(val)
            return n if n > 0 else None
        except (TypeError, ValueError):
            return None

    max_files = _parse_int_param("max_files")
    min_files = _parse_int_param("min_files")

    language = (project.config.language or "en").lower()
    if language not in ("en", "ja", "it"):
        language = "en"

    gen = generation_svc.stream_plan(
        root=root,
        model=model,
        user_prompt=user_prompt,
        folder_name=root.name,
        file_tree_text=file_tree_text,
        context_md=context_md,
        git_log=git_log,
        file_summaries=file_summaries,
        project_index_json=project_index_json,
        pinned_contents=pinned_contents or None,
        workspace_summaries=workspace_summaries or None,
        max_files=max_files,
        min_files=min_files,
        language=language,
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

    # Path safety check at approval time (defense-in-depth before any file is written)
    root_resolved = project.root.resolve()
    for f in plan.files:
        try:
            candidate = (project.root / f.path).resolve()
            if not candidate.is_relative_to(root_resolved):
                return jsonify({
                    "error": "UnsafePath",
                    "message": f"プロジェクト外を指すパスが含まれています: {f.path}",
                }), 400
        except Exception:
            return jsonify({
                "error": "InvalidPath",
                "message": f"無効なパスが含まれています: {f.path}",
            }), 400

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
    language = (project.config.language or "en").lower()
    if language not in ("en", "ja", "it"):
        language = "en"

    gen = generation_svc.stream_all_files(
        root=root,
        plan=plan,
        model=model,
        context_md=context_md,
        language=language,
    )

    def _update_ctx():
        threading.Thread(
            target=generation_svc.generate_context_md,
            args=(root, model, plan, project_svc),
            daemon=True,
        ).start()

    return _sse_response(_after_done(gen, _update_ctx))


@bp.route("/plan/saved", methods=["GET"])
def get_saved_plan():
    """
    保存済みプラン（.localforge/plan.json）を返す。

    Response JSON:
        plan: {project_name, description, approved, files: [{path, description, action, modification_notes, exists}]}
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    plan = project_svc.load_generation_plan(project.root)
    if not plan:
        return jsonify({"error": "NoPlan", "message": "保存済みプランが見つかりません"}), 404

    root = project.root
    files = []
    for f in plan.files:
        files.append({
            "path": f.path,
            "description": f.description,
            "action": f.action,
            "modification_notes": f.modification_notes,
            "exists": (root / f.path).exists(),
        })

    return jsonify({
        "plan": {
            "project_name": plan.project_name,
            "description": plan.description,
            "approved": plan.approved,
            "files": files,
        }
    })


@bp.route("/resume", methods=["GET"])
def stream_resume_generation():
    """
    前回中断されたプランの続きからSSEストリーミング生成を再開する。
    generation_log.jsonlを参照して完了済みファイルをスキップする。

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
    progress = project_svc.get_generation_progress(root)
    start_from = progress.get("start_from", 0)
    language = (project.config.language or "en").lower()
    if language not in ("en", "ja", "it"):
        language = "en"

    gen = generation_svc.stream_all_files(
        root=root,
        plan=plan,
        model=model,
        context_md=context_md,
        start_from=start_from,
        language=language,
    )

    def _update_ctx():
        threading.Thread(
            target=generation_svc.generate_context_md,
            args=(root, model, plan, project_svc),
            daemon=True,
        ).start()

    return _sse_response(_after_done(gen, _update_ctx))


@bp.route("/logs", methods=["GET"])
def get_generation_logs():
    """
    生成ログをすべて取得する。
    """
    project_svc = current_app.config["project_service"]
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    index_adapter = current_app.config["index_adapter"]
    log_path = project.root / ".localforge" / "generation_log.jsonl"

    if not log_path.exists():
        return jsonify({"logs": []})

    entries = index_adapter.load_log_entries(log_path)
    return jsonify({"logs": [e.model_dump() for e in entries]})


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
    プランにないファイルでも、プロジェクトインデックスから合成したプランで再生成できる。

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

    model = project.config.model
    if not model:
        return jsonify({"error": "NoModel", "message": "モデルが選択されていません。UIでモデルを選択してください"}), 400

    root = project.root
    context_md = project_svc.get_context_md(root)
    language = (project.config.language or "en").lower()
    if language not in ("en", "ja", "it"):
        language = "en"

    plan = project_svc.load_generation_plan(root)
    if not plan or not any(f.path == file_path for f in plan.files):
        plan = _synthesize_plan_for_file(root, file_path)

    gen = generation_svc.stream_regenerate_file(
        root=root,
        plan=plan,
        model=model,
        context_md=context_md,
        file_path=file_path,
        language=language,
    )

    def _update_ctx():
        threading.Thread(
            target=generation_svc.update_context_md_incremental,
            args=(root, model, file_path, project_svc),
            daemon=True,
        ).start()

    return _sse_response(_after_done(gen, _update_ctx))


def _synthesize_plan_for_file(root: Path, file_path: str) -> GenerationPlan:
    """
    プランが存在しない、またはファイルがプランにない場合、
    プロジェクトインデックス（JSONL）から合成したGenerationPlanを返す。
    LLMがプロジェクト全体の文脈を持った状態でファイルを再生成できるようにする。
    """
    index_adapter = current_app.config["index_adapter"]
    localforge_dir = root / ".localforge"

    chunks = []
    try:
        chunks = index_adapter.load_chunks(localforge_dir / "index.jsonl")
    except Exception:
        pass

    project_index = None
    try:
        project_index = index_adapter.load_index(localforge_dir / "project_index.json")
    except Exception:
        pass

    project_name = project_index.project_name if project_index else root.name
    project_description = project_index.summary if project_index else ""

    # 各ファイルの最初のチャンクのサマリーを使ってPlannedFileを合成
    # 既存ファイルは action="modify" に設定して誤上書きを防ぐ
    file_map: dict[str, str] = {}
    for chunk in chunks:
        if chunk.path not in file_map:
            file_map[chunk.path] = chunk.summary or chunk.path

    planned_files = [
        PlannedFile(
            path=p,
            description=s,
            action="modify" if (root / p).exists() else "create",
        )
        for p, s in file_map.items()
    ]

    # 対象ファイルがインデックスにない場合でも必ずリストに含める
    if not any(f.path == file_path for f in planned_files):
        action = "modify" if (root / file_path).exists() else "create"
        planned_files.append(PlannedFile(
            path=file_path,
            description=f"{file_path} を再生成する",
            action=action,
        ))

    return GenerationPlan(
        project_name=project_name,
        description=project_description,
        files=planned_files,
        approved=True,
    )


