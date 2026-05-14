"""
プロジェクトルート — /api/project/* エンドポイントの定義。
フォルダ選択・モード検出・ファイルツリー・設定管理を提供する。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from localforge.application.project_service import ProjectService
from localforge.domain.exceptions import LocalForgeError
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

bp = Blueprint("project", __name__, url_prefix="/api/project")


def _get_project_svc() -> ProjectService:
    """現在のアプリコンテキストからProjectServiceを取得する。"""
    return current_app.config["project_service"]


def _get_llm() -> OllamaClient:
    return current_app.config["llm"]


def _error_response(exc: Exception, status: int = 500):
    """エラーレスポンスを生成する。"""
    return jsonify({
        "error": type(exc).__name__,
        "message": str(exc),
    }), status


@bp.route("/open", methods=["POST"])
def open_project():
    """
    フォルダ選択ダイアログを表示してプロジェクトを開く。
    pywebview環境ではネイティブフォルダ選択ダイアログを使用する。
    ブラウザ環境ではリクエストボディからパスを受け取る。

    Request JSON:
        path (str, optional): フォルダの絶対パス（テスト・API用）

    Response JSON:
        mode, project_root, banner, file_tree
    """
    project_svc = _get_project_svc()
    data = request.get_json(silent=True) or {}

    # pywebviewウィンドウからのフォルダ選択
    _raw_path = data.get("path")
    folder_path: str | None = _raw_path if isinstance(_raw_path, str) else None

    if not folder_path:
        # pywebviewのウィンドウAPIを試みる
        try:
            import webview
            windows = webview.windows
            if windows:
                result = windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                if result and len(result) > 0:
                    item = result[0]
                    folder_path = item if isinstance(item, str) else None
        except (ImportError, Exception) as exc:
            logger.debug("pywebviewフォルダ選択不可: %s", exc)

    if not folder_path:
        return jsonify({"error": "NoFolderSelected", "message": "フォルダが選択されませんでした"}), 400

    root = Path(folder_path)
    if not root.is_dir():
        return jsonify({
            "error": "InvalidPath",
            "message": f"指定されたパスはディレクトリではありません: {folder_path}"
        }), 400

    try:
        project = project_svc.open_project(root)
    except LocalForgeError as exc:
        return _error_response(exc)

    llm = _get_llm()

    # プロジェクト設定の num_thread を LLM クライアントに適用する
    if project.config.num_thread is not None:
        llm.set_num_thread(project.config.num_thread)

    mode = project.mode.value
    banner_messages = {
        "generate": "新しいプロジェクトを作成します。プロンプトを入力してください。",
        "resume": "プロジェクトを再開します。",
        "explain": "既存のコードベースを検出しました — プロジェクト分析を実行します...",
    }

    def _node_to_dict(node) -> dict:
        d = {
            "name": node.name,
            "path": node.path,
            "is_dir": node.is_dir,
            "status": node.status.value,
        }
        if node.is_dir:
            d["children"] = [_node_to_dict(c) for c in node.children]
        else:
            d["size"] = node.size
        return d

    return jsonify({
        "mode": mode,
        "project_root": str(root),
        "banner": banner_messages.get(mode, ""),
        "file_tree": [_node_to_dict(n) for n in project.file_tree],
        "model": project.config.model,
    })


@bp.route("/tree", methods=["GET"])
def get_tree():
    """
    現在のプロジェクトのファイルツリーをJSON形式で返す。

    Response JSON:
        file_tree: ファイルノードの配列
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    nodes = project_svc.get_file_tree(project.root)

    def _node_to_dict(node) -> dict:
        d = {
            "name": node.name,
            "path": node.path,
            "is_dir": node.is_dir,
            "status": node.status.value,
        }
        if node.is_dir:
            d["children"] = [_node_to_dict(c) for c in node.children]
        else:
            d["size"] = node.size
        return d

    return jsonify({"file_tree": [_node_to_dict(n) for n in nodes]})


@bp.route("/status", methods=["GET"])
def get_status():
    """
    現在のプロジェクト状態（mode, root, model, git_branch）を返す。

    Response JSON:
        mode, root, model, git_branch
    """
    project_svc = _get_project_svc()
    return jsonify(project_svc.get_project_status())


@bp.route("/models", methods=["GET"])
def list_models():
    """
    Ollamaで利用可能なモデル一覧を返す。

    Response JSON:
        models: モデル名の配列
    """
    llm = _get_llm()
    try:
        models = llm.list_models()
        return jsonify({"models": models})
    except LocalForgeError as exc:
        return _error_response(exc)


@bp.route("/model", methods=["POST"])
def set_model():
    """
    使用するモデルを変更してconfig.jsonに保存する。

    Request JSON:
        model (str): 新しいモデル名

    Response JSON:
        model: 設定されたモデル名
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"error": "InvalidModel", "message": "モデル名が指定されていません"}), 400

    old_model = project.config.model
    try:
        project_svc.set_model(project.root, model)
        if old_model and old_model != model:
            _get_llm().unload_model(old_model)
        return jsonify({"model": model})
    except LocalForgeError as exc:
        return _error_response(exc)


@bp.route("/context", methods=["GET"])
def get_context():
    """
    .localforge/context.mdの内容を返す。

    Response JSON:
        content: context.mdの内容文字列
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    content = project_svc.get_context_md(project.root)
    return jsonify({"content": content})


@bp.route("/file-content", methods=["GET"])
def get_file_content():
    """
    指定ファイルのコンテンツを返す（ファイルビューワー用）。

    Query params:
        path (str): プロジェクトルートからの相対パス

    Response JSON:
        content: ファイルの内容
        path: ファイルパス
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    rel_path = request.args.get("path", "").strip()
    if not rel_path:
        return jsonify({"error": "NoPath", "message": "パスが指定されていません"}), 400

    file_path = project.root / rel_path
    # セキュリティ: プロジェクトルート外へのアクセスを拒否
    try:
        file_path.resolve().relative_to(project.root.resolve())
    except ValueError:
        return jsonify({"error": "AccessDenied", "message": "プロジェクトルート外へのアクセスは禁止されています"}), 403

    if not file_path.is_file():
        return jsonify({"error": "FileNotFound", "message": f"ファイルが見つかりません: {rel_path}"}), 404

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return jsonify({"content": content, "path": rel_path})
    except OSError as exc:
        return jsonify({"error": "ReadError", "message": str(exc)}), 500


@bp.route("/num-thread", methods=["GET"])
def get_num_thread():
    """
    現在のCPUスレッド設定とシステムのCPUコア数を返す。

    Response JSON:
        num_thread: 現在の設定（nullでデフォルト自動）
        cpu_count: システムのCPUコア数
    """
    llm = _get_llm()
    return jsonify({
        "num_thread": llm.num_thread,
        "cpu_count": os.cpu_count(),
    })


@bp.route("/num-thread", methods=["POST"])
def set_num_thread():
    """
    Ollamaが使用するCPUスレッド数を設定する。
    次回以降の全LLMコールに即座に適用される。

    Request JSON:
        num_thread (int or null): スレッド数（nullでデフォルト自動設定に戻す）

    Response JSON:
        num_thread: 設定されたスレッド数
        cpu_count: システムのCPUコア数
    """
    llm = _get_llm()
    data = request.get_json(silent=True) or {}
    num_thread = data.get("num_thread")

    if num_thread is not None:
        if not isinstance(num_thread, int) or num_thread < 1:
            return jsonify({
                "error": "InvalidValue",
                "message": "num_threadは1以上の整数を指定してください",
            }), 400
        cpu_count = os.cpu_count() or 1
        num_thread = min(num_thread, cpu_count)

    llm.set_num_thread(num_thread)

    # プロジェクトが開かれている場合は config.json にも永続化する
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if project:
        try:
            project.config.num_thread = num_thread
            project_svc.save_config(project.root, project.config)
        except Exception as exc:
            logger.warning("num_thread の config.json への保存に失敗しました: %s", exc)

    return jsonify({
        "num_thread": llm.num_thread,
        "cpu_count": os.cpu_count(),
    })


def _get_git():
    """現在のアプリコンテキストからGitAdapterを取得する。"""
    return current_app.config["git"]


@bp.route("/save-file", methods=["POST"])
def save_file():
    """
    ファイルを手動編集して保存する（バックアップ + gitコミット）。

    Request JSON:
        path (str): プロジェクトルートからの相対パス
        content (str): 新しいファイル内容

    Response JSON:
        saved: True
        path: 保存されたファイルパス
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    rel_path = data.get("path", "").strip()
    content = data.get("content")

    if not rel_path:
        return jsonify({"error": "NoPath", "message": "パスが指定されていません"}), 400
    if content is None:
        return jsonify({"error": "NoContent", "message": "コンテンツが指定されていません"}), 400

    file_path = project.root / rel_path
    try:
        file_path.resolve().relative_to(project.root.resolve())
    except ValueError:
        return jsonify({"error": "AccessDenied", "message": "プロジェクトルート外へのアクセスは禁止されています"}), 403

    try:
        # バックアップ作成
        if file_path.is_file():
            bak_path = file_path.with_suffix(file_path.suffix + ".bak")
            bak_path.write_bytes(file_path.read_bytes())
            # *.bak を .gitignore に追加
            gitignore = project.root / ".gitignore"
            if gitignore.exists():
                existing = gitignore.read_text(encoding="utf-8")
                if "*.bak" not in existing:
                    gitignore.write_text(existing.rstrip() + "\n*.bak\n", encoding="utf-8")
            else:
                gitignore.write_text("*.bak\n", encoding="utf-8")

        # ファイルを書き込む
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        # gitコミット
        git = _get_git()
        try:
            git.commit_all(project.root, f"LocalForge: {rel_path} を手動編集")
        except Exception as exc:
            logger.warning("手動編集のgitコミット失敗 (無視): %s", exc)

        return jsonify({"saved": True, "path": rel_path})
    except OSError as exc:
        return jsonify({"error": "WriteError", "message": str(exc)}), 500


@bp.route("/unload", methods=["POST"])
def unload_model():
    """
    指定されたモデルをVRAMから明示的にアンロードする。

    Request JSON:
        model (str): アンロードするモデル名

    Response JSON:
        unloaded: True
    """
    data = request.get_json(silent=True) or {}
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"error": "NoModel", "message": "モデル名が指定されていません"}), 400

    llm = _get_llm()
    try:
        llm.unload_model(model)
        return jsonify({"unloaded": True})
    except Exception as exc:
        return _error_response(exc)


@bp.route("/vram", methods=["GET"])
def get_vram():
    """
    現在のVRAM使用状況を返す。

    Response JSON:
        total, used, free (int, MiB) or None
    """
    llm = _get_llm()
    info = llm.get_vram_info()
    return jsonify(info)


@bp.route("/ollama-status", methods=["GET"])
def ollama_status():
    """
    Ollamaサーバーの状態と利用可能なモデル一覧を返す。
    フロントエンドが起動時ヘルスチェックに使用する。

    Response JSON:
        available (bool): Ollamaサーバーが接続可能かどうか
        models (list[str]): 利用可能なモデル名のリスト
        error (str|null): エラーメッセージ（利用不可の場合）
    """
    llm = _get_llm()
    if not llm.is_available():
        return jsonify({
            "available": False,
            "models": [],
            "error": "Ollamaサーバーに接続できません。Ollamaが起動しているか確認してください。",
        })
    try:
        models = llm.list_models()
        return jsonify({
            "available": True,
            "models": models,
            "error": None,
        })
    except Exception as exc:
        return jsonify({
            "available": True,
            "models": [],
            "error": f"モデル一覧の取得に失敗しました: {exc}",
        })


@bp.route("/pinned", methods=["GET"])
def get_pinned():
    """
    現在のプロジェクトのピン留めコンテキストパスを返す。

    Response JSON:
        pinned: [str]
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    return jsonify({"pinned": project_svc.get_pinned_context(project.root)})


@bp.route("/pinned", methods=["POST"])
def save_pinned():
    """
    ピン留めコンテキストパスを保存する。

    Request JSON:
        paths: [str]  プロジェクト相対パスのリスト（ファイルまたはフォルダ）

    Response JSON:
        ok, pinned
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    paths = data.get("paths", [])
    if not isinstance(paths, list):
        return jsonify({"error": "InvalidData", "message": "pathsはリストである必要があります"}), 400

    paths = [str(p) for p in paths if isinstance(p, str) and p.strip()]
    project_svc.save_pinned_context(project.root, paths)
    logger.info("ピン留めコンテキスト更新: %d件", len(paths))

    return jsonify({"ok": True, "pinned": paths})
