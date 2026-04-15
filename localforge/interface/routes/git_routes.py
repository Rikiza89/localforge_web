"""
gitルート — /api/git/* エンドポイントの定義。
git操作（init, commit, log, diff, status）を提供する。
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from localforge.application.project_service import ProjectService
from localforge.domain.exceptions import GitOperationError, LocalForgeError
from localforge.infrastructure.git_adapter import GitAdapter

logger = logging.getLogger(__name__)

bp = Blueprint("git", __name__, url_prefix="/api/git")


def _get_project_svc() -> ProjectService:
    return current_app.config["project_service"]


def _get_git() -> GitAdapter:
    return current_app.config["git"]


def _error_response(exc: Exception, status: int = 500):
    return jsonify({"error": type(exc).__name__, "message": str(exc)}), status


@bp.route("/init", methods=["POST"])
def git_init():
    """
    現在のプロジェクトディレクトリでgit initを実行する。

    Response JSON:
        initialized: true
        path: リポジトリパス
    """
    project_svc = _get_project_svc()
    git = _get_git()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    try:
        git.init(project.root)
        return jsonify({"initialized": True, "path": str(project.root)})
    except GitOperationError as exc:
        return _error_response(exc)
    except LocalForgeError as exc:
        return _error_response(exc)


@bp.route("/commit", methods=["POST"])
def git_commit():
    """
    すべての変更をステージングしてコミットする。

    Request JSON:
        message (str, optional): コミットメッセージ（省略時はデフォルトメッセージ）

    Response JSON:
        hash: コミットハッシュ
        message: コミットメッセージ
    """
    project_svc = _get_project_svc()
    git = _get_git()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    message = data.get("message", "LocalForge: 変更をコミット").strip()
    if not message:
        message = "LocalForge: 変更をコミット"

    try:
        commit_hash = git.commit_all(project.root, message)
        return jsonify({"hash": commit_hash, "message": message})
    except GitOperationError as exc:
        return _error_response(exc)


@bp.route("/log", methods=["GET"])
def git_log():
    """
    直近20コミットのログをJSON配列で返す。

    Query params:
        n (int, optional): 取得件数（デフォルト20）

    Response JSON:
        commits: [{hash, message, author, date}, ...]
    """
    project_svc = _get_project_svc()
    git = _get_git()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    try:
        n = int(request.args.get("n", 20))
    except ValueError:
        n = 20

    try:
        entries = git.get_log(project.root, max_entries=n)
        return jsonify({"commits": entries})
    except GitOperationError as exc:
        return _error_response(exc)


@bp.route("/diff", methods=["GET"])
def git_diff():
    """
    git diff HEADの結果をテキストで返す。

    Response JSON:
        diff: diff出力テキスト
    """
    project_svc = _get_project_svc()
    git = _get_git()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    try:
        diff_text = git.get_diff(project.root)
        return jsonify({"diff": diff_text})
    except GitOperationError as exc:
        return _error_response(exc)


@bp.route("/status", methods=["GET"])
def git_status():
    """
    git status --shortの結果を返す。

    Response JSON:
        status: ステータス出力テキスト
        branch: 現在のブランチ名
    """
    project_svc = _get_project_svc()
    git = _get_git()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    try:
        status_text = git.get_status(project.root)
        branch = git.get_current_branch(project.root)
        return jsonify({"status": status_text, "branch": branch})
    except GitOperationError as exc:
        return _error_response(exc)
