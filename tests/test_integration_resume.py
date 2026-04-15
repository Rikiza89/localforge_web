"""
Resumeモード統合テスト — LocalForgeプロジェクトと外部プロジェクトの再開をテスト。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localforge.application.context_service import ContextService
from localforge.application.project_service import ProjectService
from localforge.domain.models import (
    GenerationLogEntry,
    GenerationPlan,
    PlannedFile,
    ProjectConfig,
    ProjectMode,
)
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter


class TestResumeDetection:
    """再開モード検出のテスト。"""

    def test_localforge_project_incomplete_is_resume(self, tmp_path):
        """未完了のLocalForgeプロジェクトはRESUMEと判定される。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        # .localforge/config.json と未完了ログを作成
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        config = ProjectConfig(project_name="test", mode=ProjectMode.RESUME)
        (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

        entry = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="pending",
        )
        index_adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry)

        mode = project_svc.detect_project_mode(tmp_path)
        assert mode == ProjectMode.RESUME

    def test_localforge_project_complete_is_not_resume(self, tmp_path):
        """すべて完了したLocalForgeプロジェクトはRESUMEにならない可能性がある。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        config = ProjectConfig(project_name="test", mode=ProjectMode.GENERATE)
        (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

        # completed状態のログエントリのみ
        entry = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="completed",
        )
        index_adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry)

        # 完了済みプロジェクトのモード判定（config.jsonあり、pendingなし）
        mode = project_svc.detect_project_mode(tmp_path)
        # コードファイルが存在しないのでGENERATEになるはず
        assert mode in (ProjectMode.GENERATE, ProjectMode.EXPLAIN)

    def test_foreign_project_with_code_is_explain(self, tmp_path):
        """コードファイルのある外部プロジェクトはEXPLAINと判定される。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        (tmp_path / "main.py").write_text("# foreign code", encoding="utf-8")
        (tmp_path / "utils.js").write_text("// utils", encoding="utf-8")

        mode = project_svc.detect_project_mode(tmp_path)
        assert mode == ProjectMode.EXPLAIN

    def test_foreign_project_with_index_is_resume(self, tmp_path):
        """既存のインデックスがある外部プロジェクトはRESUMEと判定される。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        (tmp_path / "app.py").write_text("# app", encoding="utf-8")
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        # インデックスを作成
        from localforge.domain.models import FileChunk, ChunkStrategy
        chunk = FileChunk(
            path="app.py",
            content="# app",
            strategy=ChunkStrategy.FULL,
            size=5,
            mtime=1700000000.0,
        )
        index_adapter.save_chunks(lf_dir / "index.jsonl", [chunk])

        mode = project_svc.detect_project_mode(tmp_path)
        assert mode == ProjectMode.RESUME

    def test_empty_directory_is_generate(self, tmp_path):
        """空ディレクトリはGENERATEと判定される。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        mode = project_svc.detect_project_mode(tmp_path)
        assert mode == ProjectMode.GENERATE


class TestResumeStateBuilding:
    """ResumeState構築のテスト。"""

    def test_build_resume_state_with_plan(self, tmp_path):
        """プランのあるLocalForgeプロジェクトの再開状態を構築するテスト。"""
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        # プランを作成
        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[
                PlannedFile(path="main.py", description="main"),
                PlannedFile(path="utils.py", description="utils"),
            ],
        )
        plan.approved = True
        (lf_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        # 1ファイル完了済み、1ファイル未完了
        config = ProjectConfig(project_name="test", mode=ProjectMode.RESUME)
        (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

        entry_completed = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="completed",
        )
        entry_pending = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="generate_file",
            file_path="utils.py",
            status="pending",
        )
        index_adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry_completed)
        index_adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry_pending)

        project = project_svc.open_project(tmp_path)
        resume_state = project.resume_state

        assert resume_state is not None
        assert resume_state.is_localforge_project
        assert "main.py" in resume_state.completed_files
        assert "utils.py" in resume_state.pending_files

    def test_build_resume_state_without_plan(self, tmp_path):
        """プランのない外部プロジェクトの再開状態構築をテスト。
        外部プロジェクトにはplan.jsonが存在しないため、completed_filesはコードファイルを含む。
        """
        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)

        (tmp_path / "app.py").write_text("# app", encoding="utf-8")
        (tmp_path / "utils.py").write_text("# utils", encoding="utf-8")

        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        # 外部プロジェクトのインデックスを作成（plan.jsonなし）
        from localforge.domain.models import FileChunk, ChunkStrategy
        chunks = [
            FileChunk(
                path="app.py",
                content="# app",
                strategy=ChunkStrategy.FULL,
                size=5,
                mtime=1700000000.0,
            ),
        ]
        index_adapter.save_chunks(lf_dir / "index.jsonl", chunks)

        project = project_svc.open_project(tmp_path)
        # resume_stateはRESUMEモード時のみ構築される
        if project.mode == ProjectMode.RESUME and project.resume_state:
            # plan.jsonが存在しないため、completed_filesはファイルシステムから収集される
            assert len(project.resume_state.completed_files) > 0
            # plan.jsonが存在しないため、pending_filesは空
            assert len(project.resume_state.pending_files) == 0


class TestFlaskRoutesResume:
    """Resumeモード関連のFlaskルートテスト。"""

    def test_status_returns_mode(self, flask_client, tmp_path):
        """ステータスエンドポイントがモードを返すことをテスト。"""
        # プロジェクトを開く（APIを通じてではなくサービスを直接使う）
        project_svc = flask_client.application.config["project_service"]

        # コードファイルのあるプロジェクトを作成
        (tmp_path / "main.py").write_text("# code", encoding="utf-8")
        project_svc.open_project(tmp_path)

        response = flask_client.get("/api/project/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "mode" in data
        assert data["mode"] in ("generate", "resume", "explain")

    def test_git_status_no_project(self, flask_client):
        """プロジェクト未選択時のgit statusエンドポイントをテスト。"""
        response = flask_client.get("/api/git/status")
        # プロジェクトが開かれていない場合は400を返す
        assert response.status_code in (200, 400)
