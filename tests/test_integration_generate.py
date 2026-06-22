"""
Generateパイプライン統合テスト — エンドツーエンドの生成フローをテストする。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localforge.application.context_service import ContextService
from localforge.application.generation_service import GenerationService, reset_cancel
from localforge.application.project_service import ProjectService
from localforge.domain.models import GenerationPlan, PlannedFile, ProjectMode
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient


class TestGeneratePipelineE2E:
    """Generateパイプラインのエンドツーエンドテスト。"""

    @pytest.fixture
    def gen_env(self, tmp_path):
        """テスト環境を構築するフィクスチャ。"""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.is_available.return_value = True
        mock_llm.list_models.return_value = ["llama3.2"]

        fs = FileSystemAdapter()
        git = GitAdapter()
        index_adapter = IndexAdapter()
        context = ContextService()

        project_svc = ProjectService(fs=fs, git=git, index=index_adapter)
        gen_svc = GenerationService(
            fs=fs,
            git=git,
            index_adapter=index_adapter,
            llm=mock_llm,
            context=context,
        )

        return {
            "root": tmp_path,
            "project_svc": project_svc,
            "gen_svc": gen_svc,
            "mock_llm": mock_llm,
        }

    def test_full_generate_pipeline(self, gen_env):
        """プラン生成 → 承認 → ファイル生成の完全フローをテスト。"""
        root = gen_env["root"]
        project_svc = gen_env["project_svc"]
        gen_svc = gen_env["gen_svc"]
        mock_llm = gen_env["mock_llm"]

        reset_cancel()

        # 1. プロジェクトを開く
        project = project_svc.open_project(root)
        assert project.mode == ProjectMode.GENERATE

        # 2. プラン生成 (LLMモック)
        plan_json = json.dumps({
            "project_name": "hello_app",
            "description": "シンプルなHello Worldアプリ",
            "files": [
                {"path": "main.py", "description": "エントリーポイント", "dependencies": []},
                {"path": "utils.py", "description": "ユーティリティ", "dependencies": ["main.py"]},
            ]
        })

        def mock_plan_stream(model, prompt, system=None, **kwargs):
            yield plan_json

        mock_llm.stream_completion.side_effect = mock_plan_stream

        plan_events = list(gen_svc.stream_plan(
            root=root,
            model="llama3.2",
            user_prompt="Hello Worldアプリを作ってください",
            folder_name="hello_app",
            file_tree_text="",
            context_md="",
            git_log="",
        ))

        assert any(e.get("done") for e in plan_events)

        # 3. プランをパース・承認
        plan = gen_svc.parse_plan(plan_json)
        assert plan.project_name == "hello_app"
        assert len(plan.files) == 2

        plan.approved = True
        project_svc.save_generation_plan(root, plan)

        # 4. ファイル生成
        file_counter = [0]

        def mock_file_stream(model, prompt, system=None, **kwargs):
            file_counter[0] += 1
            yield f"# Generated file {file_counter[0]}\nprint('hello')\n"

        mock_llm.stream_completion.side_effect = mock_file_stream

        gen_events = list(gen_svc.stream_all_files(
            root=root,
            plan=plan,
            model="llama3.2",
            context_md="",
        ))

        # ファイルが書き込まれたことを確認
        file_written_events = [e for e in gen_events if "file_written" in e]
        assert len(file_written_events) == 2

        # 実際にファイルが存在することを確認
        assert (root / "main.py").exists()
        assert (root / "utils.py").exists()

        # 生成ログが記録されたことを確認
        log_path = root / ".localforge" / "generation_log.jsonl"
        assert log_path.exists()

        # gitコミットが作成されたことを確認（git initが実行済みなら）
        if (root / ".git").is_dir():
            git = GitAdapter()
            entries = git.get_log(root, max_entries=10)
            assert len(entries) >= 1

    def test_single_file_regeneration(self, gen_env):
        """単一ファイルの再生成（既存ファイル → diffモード）をテスト。"""
        root = gen_env["root"]
        gen_svc = gen_env["gen_svc"]
        mock_llm = gen_env["mock_llm"]

        reset_cancel()

        # 既存ファイルを作成（既存ファイルは SEARCH/REPLACE diff モードで編集される）
        (root / "main.py").write_text("# old content", encoding="utf-8")

        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[PlannedFile(path="main.py", description="メイン", dependencies=[])],
        )

        def mock_stream(model, prompt, system=None, **kwargs):
            yield (
                "<<<<<<< SEARCH\n"
                "# old content\n"
                "=======\n"
                "# regenerated content\n"
                "print('new')\n"
                ">>>>>>> REPLACE\n"
            )

        mock_llm.stream_completion.side_effect = mock_stream

        events = list(gen_svc.stream_regenerate_file(
            root=root,
            plan=plan,
            model="llama3.2",
            context_md="",
            file_path="main.py",
        ))

        done_events = [e for e in events if e.get("done")]
        file_written_events = [e for e in events if "file_written" in e]

        assert len(done_events) == 1
        assert len(file_written_events) == 1
        assert "regenerated" in (root / "main.py").read_text(encoding="utf-8")

    def test_single_file_regeneration_create_path(self, gen_env):
        """単一ファイルの再生成（ファイル未存在 → 全体生成モード）をテスト。"""
        root = gen_env["root"]
        gen_svc = gen_env["gen_svc"]
        mock_llm = gen_env["mock_llm"]

        reset_cancel()

        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[PlannedFile(path="newfile.py", description="新規", dependencies=[])],
        )

        def mock_stream(model, prompt, system=None, **kwargs):
            yield "# generated content\nprint('new')\n"

        mock_llm.stream_completion.side_effect = mock_stream

        events = list(gen_svc.stream_regenerate_file(
            root=root,
            plan=plan,
            model="llama3.2",
            context_md="",
            file_path="newfile.py",
        ))

        done_events = [e for e in events if e.get("done")]
        assert len(done_events) == 1
        assert "generated" in (root / "newfile.py").read_text(encoding="utf-8")

    def test_cancel_generation(self, gen_env):
        """生成キャンセルをテスト。"""
        from localforge.application.generation_service import request_cancel

        root = gen_env["root"]
        gen_svc = gen_env["gen_svc"]
        mock_llm = gen_env["mock_llm"]

        reset_cancel()

        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[
                PlannedFile(path=f"file_{i}.py", description=f"file {i}", dependencies=[])
                for i in range(5)
            ],
        )

        call_count = [0]

        def mock_stream(model, prompt, system=None, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                request_cancel()
            yield "# code\n"

        mock_llm.stream_completion.side_effect = mock_stream

        events = list(gen_svc.stream_all_files(
            root=root,
            plan=plan,
            model="llama3.2",
            context_md="",
        ))

        cancel_events = [e for e in events if "error" in e and "キャンセル" in e.get("error", "")]
        assert len(cancel_events) >= 1
