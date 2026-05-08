"""
アプリケーション層の単体テスト — サービスクラスのテスト。
LLMとファイルシステムはモックを使用する。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localforge.application.context_service import ContextService, _estimate_tokens
from localforge.application.generation_service import GenerationService, reset_cancel
from localforge.application.project_service import ProjectService
from localforge.application.analysis_service import AnalysisService, _extract_python_landmarks
from localforge.domain.exceptions import PlanParseError
from localforge.domain.models import (
    ChunkStrategy,
    FileChunk,
    GenerationPlan,
    PlannedFile,
    ProjectMode,
)


class TestContextService:
    """ContextServiceの単体テスト。"""

    def test_estimate_tokens(self):
        text = "hello world foo bar"
        estimated = _estimate_tokens(text)
        assert estimated > 0
        assert isinstance(estimated, int)

    def test_build_plan_prompt(self, context_service):
        prompt, tokens = context_service.build_plan_prompt(
            user_prompt="Todoアプリを作って",
            folder_name="todo_app",
            file_tree_text="",
            context_md="",
            git_log="",
        )
        assert "Todoアプリを作って" in prompt
        assert "todo_app" in prompt
        assert "JSON" in prompt

    def test_build_plan_prompt_includes_context(self, context_service):
        prompt, tokens = context_service.build_plan_prompt(
            user_prompt="test",
            folder_name="proj",
            file_tree_text="├── main.py",
            context_md="# Context\nThis is a web app",
            git_log="abc1234 Initial commit",
        )
        assert "main.py" in prompt
        assert "Context" in prompt
        assert "abc1234" in prompt

    def test_build_file_generation_prompt(self, context_service):
        prompt, tokens = context_service.build_file_generation_prompt(
            target_file="src/main.py",
            target_description="エントリーポイント",
            context_md="",
            plan_json='{"files": []}',
            dependency_contents=[],
        )
        assert "src/main.py" in prompt
        assert "エントリーポイント" in prompt

    def test_build_file_summary_prompt(self, context_service):
        prompt = context_service.build_file_summary_prompt(
            file_path="utils.py",
            content="def helper(): pass",
            extension=".py",
        )
        assert "utils.py" in prompt
        assert "helper" in prompt

    def test_build_qa_prompt(self, context_service):
        from localforge.domain.models import Message
        prompt, tokens = context_service.build_qa_prompt(
            question="このプロジェクトは何ですか？",
            project_index_json='{"summary": "test"}',
            top_summaries=[("main.py", "エントリーポイント")],
            full_contents=[],
            conversation_history=[],
        )
        assert "このプロジェクトは何ですか？" in prompt
        assert "main.py" in prompt

    def test_token_limit_update(self):
        svc = ContextService(token_limit=1000)
        svc.update_token_limit(2000)
        assert svc._token_limit == 2000

    def test_build_report_section_prompt(self, context_service):
        prompt, tokens = context_service.build_report_section_prompt(
            section_name="Project Overview",
            project_index_json='{"summary": "web app"}',
            relevant_summaries=[("app.py", "Flaskアプリ")],
        )
        assert "Project Overview" in prompt
        assert "app.py" in prompt


class TestProjectService:
    """ProjectServiceの単体テスト。"""

    def test_detect_mode_generate_empty_dir(self, project_service, tmp_path):
        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.GENERATE

    def test_detect_mode_explain_with_code(self, project_service, tmp_path):
        (tmp_path / "main.py").write_text("# code", encoding="utf-8")
        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.EXPLAIN

    def test_detect_mode_generate_non_code_files(self, project_service, tmp_path):
        (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.GENERATE

    def test_detect_mode_resume_with_localforge(self, project_service, tmp_path):
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()
        # config.jsonを作成
        from localforge.domain.models import ProjectConfig
        config = ProjectConfig(project_name="test", mode=ProjectMode.RESUME)
        (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
        # 不完全なgeneration_log.jsonlを作成
        from localforge.domain.models import GenerationLogEntry
        from localforge.infrastructure.index_adapter import IndexAdapter
        adapter = IndexAdapter()
        entry = GenerationLogEntry(
            mode="generate", model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="pending",
        )
        adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry)

        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.RESUME

    def test_open_project_sets_current(self, project_service, tmp_path):
        project = project_service.open_project(tmp_path)
        assert project_service.current_project is not None
        assert project_service.current_project.root == tmp_path

    def test_get_context_md_empty(self, project_service, tmp_path):
        content = project_service.get_context_md(tmp_path)
        assert content == ""

    def test_save_and_get_context_md(self, project_service, tmp_path):
        project_service.save_context_md(tmp_path, "# Context\nTest content")
        content = project_service.get_context_md(tmp_path)
        assert "Context" in content

    def test_save_and_load_generation_plan(self, project_service, tmp_path, sample_plan):
        project_service.save_generation_plan(tmp_path, sample_plan)
        loaded = project_service.load_generation_plan(tmp_path)
        assert loaded is not None
        assert loaded.project_name == sample_plan.project_name
        assert len(loaded.files) == len(sample_plan.files)

    def test_get_project_status_no_project(self, project_service):
        status = project_service.get_project_status()
        assert status["mode"] is None
        assert status["root"] is None

    def test_set_model(self, project_service, tmp_path):
        project_service.open_project(tmp_path)
        project_service.set_model(tmp_path, "codellama")
        assert project_service.current_project.config.model == "codellama"


class TestGenerationService:
    """GenerationServiceの単体テスト。"""

    def test_parse_plan_valid_json(self, generation_service):
        plan_text = json.dumps({
            "project_name": "my_app",
            "description": "Test app",
            "files": [
                {"path": "main.py", "description": "Entry point", "dependencies": []},
            ]
        })
        plan = generation_service.parse_plan(plan_text)
        assert plan.project_name == "my_app"
        assert len(plan.files) == 1

    def test_parse_plan_with_code_block(self, generation_service):
        plan_text = """Here's the plan:
```json
{
  "project_name": "test_app",
  "description": "Test",
  "files": [{"path": "app.py", "description": "Main app", "dependencies": []}]
}
```"""
        plan = generation_service.parse_plan(plan_text)
        assert plan.project_name == "test_app"

    def test_parse_plan_invalid_json_raises(self, generation_service):
        with pytest.raises(PlanParseError):
            generation_service.parse_plan("not valid json at all {{{")

    def test_stream_plan_yields_tokens(self, generation_service, tmp_path):
        def fake_stream(model, prompt, system=None):
            yield '{"project_name": "app", "description": "test", "files": []}'

        generation_service._llm.stream_completion.side_effect = fake_stream

        project = MagicMock()
        project.root = tmp_path
        project.config.model = "llama3.2"

        events = list(generation_service.stream_plan(
            root=tmp_path,
            model="llama3.2",
            user_prompt="make a todo app",
            folder_name="todo",
            file_tree_text="",
            context_md="",
            git_log="",
        ))

        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]
        assert len(token_events) > 0
        assert len(done_events) == 1

    def test_stream_all_files_yields_progress(self, generation_service, tmp_path):
        reset_cancel()

        def fake_stream(model, prompt, system=None):
            yield "# generated code\n"

        generation_service._llm.stream_completion.side_effect = fake_stream

        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[
                PlannedFile(path="main.py", description="main"),
            ],
        )

        events = list(generation_service.stream_all_files(
            root=tmp_path,
            plan=plan,
            model="llama3.2",
            context_md="",
        ))

        progress_events = [e for e in events if "progress" in e]
        file_written_events = [e for e in events if "file_written" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(progress_events) >= 1
        assert len(file_written_events) >= 1
        assert len(done_events) == 1


class TestAnalysisService:
    """AnalysisServiceの単体テスト。"""

    def test_read_file_chunk_full(self, analysis_service, tmp_path):
        file_path = tmp_path / "small.py"
        content = "def hello():\n    pass\n"
        file_path.write_text(content, encoding="utf-8")

        chunk = analysis_service.read_file_chunk(file_path, tmp_path)
        assert chunk.path == "small.py"
        assert chunk.strategy == ChunkStrategy.FULL
        assert chunk.content == content

    def test_read_file_chunk_hybrid(self, analysis_service, tmp_path):
        file_path = tmp_path / "large.py"
        lines = [f"# line {i}" for i in range(250)]
        content = "\n".join(lines)
        file_path.write_text(content, encoding="utf-8")

        chunk = analysis_service.read_file_chunk(file_path, tmp_path)
        assert chunk.strategy == ChunkStrategy.HYBRID
        assert "省略" in chunk.content or "..." in chunk.content

    def test_extract_python_landmarks(self):
        source = """
import os
from pathlib import Path

class MyClass:
    def method(self):
        pass

def standalone_func():
    pass
"""
        landmarks = _extract_python_landmarks(source)
        assert len(landmarks) > 0
        # import文またはclass/function定義が含まれることを確認
        landmark_text = " ".join(landmarks)
        assert any(kw in landmark_text for kw in ["import", "class", "def"])

    def test_get_top_chunks_by_keywords(self, analysis_service, sample_chunk):
        chunks = [
            sample_chunk,
            FileChunk(
                path="auth.py",
                content="def login(): pass",
                strategy=ChunkStrategy.FULL,
                size=18,
                mtime=1700000000.0,
                summary="認証機能",
            ),
        ]
        top = analysis_service.get_top_chunks_by_keywords(chunks, "login auth", top_n=1)
        assert len(top) == 1
        assert top[0].path == "auth.py"

    def test_build_index_on_fixture(self, analysis_service, python_fixture_project):
        """15ファイルのフィクスチャでインデックス構築をテスト（Ollama不使用）。"""
        events = list(analysis_service.build_index(
            root=python_fixture_project,
            model="llama3.2",
        ))

        progress_events = [e for e in events if "progress" in e]
        done_events = [e for e in events if e.get("done")]
        assert len(progress_events) >= 1
        assert len(done_events) == 1

        # インデックスファイルが生成されたことを確認
        index_path = python_fixture_project / ".localforge" / "index.jsonl"
        assert index_path.exists()

    def test_build_index_no_status_events_without_vector(self, analysis_service, python_fixture_project):
        """VectorAdapterなしの場合、statusイベントが生成されないことをテスト。"""
        events = list(analysis_service.build_index(
            root=python_fixture_project,
            model="llama3.2",
        ))
        status_events = [e for e in events if "status" in e]
        # vector=None なので埋め込みフェーズは実行されずstatusイベントは0件
        assert len(status_events) == 0

    def test_build_index_emits_status_events_with_vector(
        self, analysis_service_with_vector, mock_vector, python_fixture_project
    ):
        """VectorAdapterありの場合、statusイベントが生成されることをテスト。"""
        events = list(analysis_service_with_vector.build_index(
            root=python_fixture_project,
            model="llama3.2",
        ))
        status_events = [e for e in events if "status" in e]
        done_events = [e for e in events if e.get("done")]
        # 埋め込みフェーズのstatusイベントが存在する
        assert len(status_events) >= 1
        assert len(done_events) == 1
        # upsert_chunkが呼び出されたことを確認
        assert mock_vector.upsert_chunk.called

    def test_get_top_chunks_semantic_fallback_to_keywords(self, analysis_service):
        """vector=Noneの場合、get_top_chunks_semanticがキーワード検索にフォールバックすることをテスト。"""
        chunks = [
            FileChunk(
                path="auth.py",
                content="def login(): pass",
                strategy=ChunkStrategy.FULL,
                size=18,
                mtime=1700000000.0,
                summary="認証・ログイン機能",
            ),
            FileChunk(
                path="utils.py",
                content="def format_date(): pass",
                strategy=ChunkStrategy.FULL,
                size=25,
                mtime=1700000001.0,
                summary="日付フォーマットユーティリティ",
            ),
        ]
        # vector=None なのでキーワード検索にフォールバック
        top = analysis_service.get_top_chunks_semantic(chunks, "login authentication", top_n=1)
        assert len(top) == 1
        assert top[0].path == "auth.py"

    def test_get_top_chunks_semantic_with_vector(self, analysis_service_with_vector, mock_vector, sample_chunk):
        """VectorAdapterありの場合、セマンティック検索が呼び出されることをテスト。"""
        mock_vector.get_top_chunks_semantic.return_value = [sample_chunk]
        chunks = [sample_chunk]
        top = analysis_service_with_vector.get_top_chunks_semantic(chunks, "hello", top_n=1)
        assert len(top) == 1
        mock_vector.get_top_chunks_semantic.assert_called_once()


class TestResumeDetection:
    """再開検出のテスト。"""

    def test_resume_incomplete_localforge_project(self, project_service, tmp_path):
        """未完了のLocalForgeプロジェクトはRESUMEモードと判定される。"""
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()

        from localforge.domain.models import ProjectConfig, GenerationLogEntry
        from localforge.infrastructure.index_adapter import IndexAdapter

        config = ProjectConfig(project_name="test", mode=ProjectMode.RESUME)
        (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

        # pending状態のログエントリを作成
        adapter = IndexAdapter()
        entry = GenerationLogEntry(
            mode="generate", model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="pending",
        )
        adapter.append_log_entry(lf_dir / "generation_log.jsonl", entry)

        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.RESUME

    def test_resume_foreign_project_with_index(self, project_service, tmp_path):
        """インデックスがある外部プロジェクトはRESUMEモードと判定される。"""
        (tmp_path / "app.py").write_text("# app", encoding="utf-8")
        lf_dir = tmp_path / ".localforge"
        lf_dir.mkdir()
        (lf_dir / "index.jsonl").write_text(
            '{"path": "app.py", "content": "# app", "strategy": "full", "size": 5, "mtime": 1700000000.0}\n',
            encoding="utf-8"
        )

        mode = project_service.detect_project_mode(tmp_path)
        assert mode == ProjectMode.RESUME
