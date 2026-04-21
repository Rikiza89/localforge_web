"""
Flaskルートの単体テスト — APIエンドポイントの動作確認。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestProjectRoutes:
    """プロジェクトルートのテスト。"""

    def test_models_endpoint_returns_list(self, flask_client):
        """モデル一覧エンドポイントがリストを返すことをテスト。"""
        # LLMモックを設定
        mock_llm = flask_client.application.config["llm"]
        mock_llm.list_models = MagicMock(return_value=["llama3.2", "codellama"])

        response = flask_client.get("/api/project/models")
        assert response.status_code == 200
        data = response.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_status_endpoint(self, flask_client):
        """ステータスエンドポイントが正しい形式のデータを返すことをテスト。"""
        response = flask_client.get("/api/project/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "mode" in data
        assert "root" in data
        assert "model" in data
        assert "git_branch" in data

    def test_context_endpoint_no_project(self, flask_client):
        """プロジェクト未選択時のコンテキストエンドポイントをテスト。"""
        response = flask_client.get("/api/project/context")
        # プロジェクトが開かれていない場合は400を返す
        assert response.status_code in (200, 400)

    def test_context_endpoint_with_project(self, flask_client, tmp_path):
        """プロジェクト選択時のコンテキストエンドポイントをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/project/context")
        assert response.status_code == 200
        data = response.get_json()
        assert "content" in data
        assert isinstance(data["content"], str)

    def test_tree_endpoint_with_project(self, flask_client, tmp_path):
        """ファイルツリーエンドポイントをテスト。"""
        (tmp_path / "main.py").write_text("# code", encoding="utf-8")
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/project/tree")
        assert response.status_code == 200
        data = response.get_json()
        assert "file_tree" in data
        assert isinstance(data["file_tree"], list)

    def test_set_model_endpoint(self, flask_client, tmp_path):
        """モデル変更エンドポイントをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.post(
            "/api/project/model",
            json={"model": "codellama"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["model"] == "codellama"

    def test_set_model_no_project(self, flask_client):
        """プロジェクト未選択時のモデル変更をテスト。"""
        # 新しいFlaskクライアントを使って現在のプロジェクトをリセット
        response = flask_client.post(
            "/api/project/model",
            json={"model": "codellama"},
        )
        # プロジェクトが開かれていない場合は400を返す
        assert response.status_code in (200, 400)

    def test_file_content_endpoint(self, flask_client, tmp_path):
        """ファイルコンテンツエンドポイントをテスト。"""
        (tmp_path / "main.py").write_text("# main code", encoding="utf-8")
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/project/file-content?path=main.py")
        assert response.status_code == 200
        data = response.get_json()
        assert "content" in data
        assert "main code" in data["content"]

    def test_file_content_path_traversal_blocked(self, flask_client, tmp_path):
        """パストラバーサル攻撃がブロックされることをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/project/file-content?path=../../etc/passwd")
        assert response.status_code in (403, 404)


class TestGenerationRoutes:
    """生成ルートのテスト。"""

    def test_approve_plan_valid(self, flask_client, tmp_path):
        """有効なプランの承認をテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        plan_data = {
            "project_name": "test_app",
            "description": "Test",
            "files": [
                {"path": "main.py", "description": "Entry point", "dependencies": []},
            ]
        }

        response = flask_client.post(
            "/api/generate/approve",
            json={"plan_json": json.dumps(plan_data)},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "plan" in data
        assert data["plan"]["file_count"] == 1

    def test_approve_plan_invalid_json(self, flask_client, tmp_path):
        """無効なJSONでのプラン承認が400を返すことをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.post(
            "/api/generate/approve",
            json={"plan_json": "invalid json {{{"},
        )
        assert response.status_code == 400

    def test_cancel_endpoint(self, flask_client):
        """キャンセルエンドポイントをテスト。"""
        response = flask_client.post("/api/generate/cancel")
        assert response.status_code == 200
        data = response.get_json()
        assert data["cancelled"] is True


class TestGitRoutes:
    """gitルートのテスト。"""

    def test_git_log_no_project(self, flask_client):
        """プロジェクト未選択時のgit logをテスト。"""
        response = flask_client.get("/api/git/log")
        assert response.status_code in (200, 400)

    def test_git_log_with_project(self, flask_client, tmp_path):
        """gitログエンドポイントをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/git/log")
        assert response.status_code == 200
        data = response.get_json()
        assert "commits" in data
        assert isinstance(data["commits"], list)

    def test_git_init_endpoint(self, flask_client, tmp_path):
        """git initエンドポイントをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.post("/api/git/init")
        assert response.status_code == 200
        data = response.get_json()
        assert data.get("initialized") is True

    def test_git_status_endpoint(self, flask_client, tmp_path):
        """git statusエンドポイントをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/git/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "branch" in data


class TestExplainRoutes:
    """説明ルートのテスト。"""

    def test_summary_no_index(self, flask_client, tmp_path):
        """インデックスなし時のサマリーエンドポイントが404を返すことをテスト。"""
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/explain/summary")
        assert response.status_code == 404

    def test_summary_with_index(self, flask_client, tmp_path):
        """インデックスありのサマリーエンドポイントをテスト。"""
        from localforge.domain.models import ProjectIndex
        from localforge.infrastructure.index_adapter import IndexAdapter

        # ProjectIndexを作成
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()
        index = ProjectIndex(
            project_root=str(tmp_path),
            project_name="test_project",
            summary="テストプロジェクトの概要",
            total_files=5,
            indexed_files=5,
        )
        adapter = IndexAdapter()
        adapter.save_index(lf_dir / "project_index.json", index)

        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/explain/summary")
        assert response.status_code == 200
        data = response.get_json()
        assert data["project_name"] == "test_project"
        assert "summary" in data

    def test_summary_includes_rag_ready_field(self, flask_client, tmp_path):
        """サマリーレスポンスにrag_readyフィールドが含まれることをテスト。"""
        from localforge.domain.models import ProjectIndex
        from localforge.infrastructure.index_adapter import IndexAdapter

        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()
        index = ProjectIndex(
            project_root=str(tmp_path),
            project_name="rag_test",
            summary="RAGテスト",
            total_files=3,
            indexed_files=3,
        )
        IndexAdapter().save_index(lf_dir / "project_index.json", index)

        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/explain/summary")
        assert response.status_code == 200
        data = response.get_json()
        assert "rag_ready" in data
        assert isinstance(data["rag_ready"], bool)

    def test_main_page_returns_html(self, flask_client):
        """メインページがHTMLを返すことをテスト。"""
        response = flask_client.get("/")
        assert response.status_code == 200
        assert b"LocalForge" in response.data


class TestProjectOllamaStatus:
    """Ollamaステータスエンドポイントのテスト。"""

    def test_ollama_status_endpoint_returns_available_field(self, flask_client):
        """ollama-statusエンドポイントがavailableフィールドを返すことをテスト。"""
        llm = flask_client.application.config["llm"]
        llm.is_available = MagicMock(return_value=True)
        llm.list_models = MagicMock(return_value=["llama3.2"])

        response = flask_client.get("/api/project/ollama-status")
        assert response.status_code == 200
        data = response.get_json()
        assert "available" in data
        assert isinstance(data["available"], bool)

    def test_ollama_status_endpoint_when_available(self, flask_client):
        """Ollamaが利用可能な場合のステータスをテスト。"""
        llm = flask_client.application.config["llm"]
        llm.is_available = MagicMock(return_value=True)
        llm.list_models = MagicMock(return_value=["llama3.2", "codellama"])

        response = flask_client.get("/api/project/ollama-status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["available"] is True
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_ollama_status_endpoint_when_unavailable(self, flask_client):
        """Ollamaが利用不可の場合のステータスをテスト。"""
        llm = flask_client.application.config["llm"]
        llm.is_available = MagicMock(return_value=False)
        llm.list_models = MagicMock(side_effect=Exception("Connection refused"))

        response = flask_client.get("/api/project/ollama-status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["available"] is False
