"""
ワークスペースルート — /api/workspace/* エンドポイントの定義。
複数プロジェクトのワークスペース管理を提供する。
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from localforge.application.analysis_service import AnalysisService
from localforge.application.project_service import ProjectService

logger = logging.getLogger(__name__)

bp = Blueprint("workspace", __name__, url_prefix="/api/workspace")


def _get_project_svc() -> ProjectService:
    return current_app.config["project_service"]


def _get_analysis_svc() -> AnalysisService:
    return current_app.config["analysis_service"]


@bp.route("/list", methods=["GET"])
def list_workspace():
    """
    現在のワークスペースに属するプロジェクト一覧を返す。
    自動検出（サブプロジェクト）と手動追加を統合して返す。

    Response JSON:
        projects: [{root, name, auto, indexed}]
    """
    project_svc = _get_project_svc()
    analysis_svc = _get_analysis_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    root = project.root
    workspace_roots = project_svc.get_workspace_roots(root)

    # 自動検出フォルダを判定（手動エントリとの比較用）
    state = project_svc.load_workspace(root)
    manual_roots = {e.root for e in state.manual_entries}

    projects = []
    for proj_path, proj_name in workspace_roots:
        idx = analysis_svc.load_project_index(proj_path)
        projects.append({
            "root": str(proj_path),
            "name": proj_name,
            "auto": str(proj_path) not in manual_roots,
            "indexed": idx is not None,
            "file_count": idx.total_files if idx else 0,
        })

    return jsonify({"projects": projects})


@bp.route("/add", methods=["POST"])
def add_project():
    """
    外部プロジェクトをワークスペースに手動追加する。

    Request JSON:
        path (str): 追加するプロジェクトの絶対パス

    Response JSON:
        ok, projects
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "NoPath", "message": "pathが指定されていません"}), 400

    proj_path = Path(raw_path)
    if not proj_path.is_dir():
        return jsonify({"error": "InvalidPath", "message": f"ディレクトリが存在しません: {raw_path}"}), 400

    if proj_path == project.root:
        return jsonify({"error": "SameProject", "message": "アクティブプロジェクト自身は追加できません"}), 400

    project_svc.add_to_workspace(project.root, proj_path)
    logger.info("ワークスペースにプロジェクトを追加: %s", proj_path)

    return jsonify({"ok": True, "name": proj_path.name, "root": str(proj_path)})


@bp.route("/remove", methods=["POST"])
def remove_project():
    """
    手動追加されたプロジェクトをワークスペースから削除する。

    Request JSON:
        path (str): 削除するプロジェクトの絶対パス

    Response JSON:
        ok
    """
    project_svc = _get_project_svc()
    project = project_svc.current_project
    if not project:
        return jsonify({"error": "NoProject", "message": "プロジェクトが開かれていません"}), 400

    data = request.get_json(silent=True) or {}
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "NoPath", "message": "pathが指定されていません"}), 400

    project_svc.remove_from_workspace(project.root, Path(raw_path))
    logger.info("ワークスペースからプロジェクトを削除: %s", raw_path)

    return jsonify({"ok": True})
