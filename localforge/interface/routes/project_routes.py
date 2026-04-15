"""
プロジェクトルート — /api/project/* エンドポイントの定義。
フォルダ選択・モード検出・ファイルツリー・設定管理を提供する。
"""

from __future__ import annotations

import logging
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
    """現在のアプリコンテキストからOllamaClientを取得する。"""
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
    folder_path: str | None = data.get("path")

    if not folder_path:
        # pywebviewのウィンドウAPIを試みる
        try:
            import webview
            windows = webview.windows
            if windows:
                result = windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                if result and len(result) > 0:
                    folder_path = result[0]
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

    try:
        project_svc.set_model(project.root, model)
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
