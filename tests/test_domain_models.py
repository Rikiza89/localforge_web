"""
ドメインモデルの単体テスト — models.py, exceptions.py のテスト。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from localforge.domain.exceptions import (
    FileWriteError,
    GitOperationError,
    IndexBuildError,
    LocalForgeError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    PlanParseError,
    ResumeStateCorruptError,
    TokenBudgetExceededWarning,
)
from localforge.domain.models import (
    ChunkStrategy,
    FileChunk,
    FileNode,
    FileStatus,
    GenerationPlan,
    GenerationLogEntry,
    Message,
    PlannedFile,
    Project,
    ProjectConfig,
    ProjectIndex,
    ProjectMode,
    ResumeState,
)
from pathlib import Path


class TestProjectMode:
    """ProjectModeの列挙型テスト。"""

    def test_values(self):
        assert ProjectMode.GENERATE == "generate"
        assert ProjectMode.RESUME == "resume"
        assert ProjectMode.EXPLAIN == "explain"

    def test_string_comparison(self):
        assert ProjectMode.GENERATE.value == "generate"


class TestFileNode:
    """FileNodeモデルのテスト。"""

    def test_basic_file_node(self):
        node = FileNode(name="main.py", path="main.py", is_dir=False)
        assert node.name == "main.py"
        assert not node.is_dir
        assert node.status == FileStatus.UNINDEXED
        assert node.children == []

    def test_directory_node_with_children(self):
        child = FileNode(name="utils.py", path="src/utils.py", is_dir=False)
        parent = FileNode(
            name="src",
            path="src",
            is_dir=True,
            children=[child],
        )
        assert parent.is_dir
        assert len(parent.children) == 1
        assert parent.children[0].name == "utils.py"

    def test_status_values(self):
        for status in FileStatus:
            node = FileNode(name="f.py", path="f.py", is_dir=False, status=status)
            assert node.status == status


class TestGenerationPlan:
    """GenerationPlanモデルのテスト。"""

    def test_basic_plan(self):
        plan = GenerationPlan(
            project_name="my_app",
            description="My app description",
            files=[
                PlannedFile(path="main.py", description="Entry point"),
                PlannedFile(path="utils.py", description="Utilities", dependencies=["main.py"]),
            ],
        )
        assert plan.project_name == "my_app"
        assert len(plan.files) == 2
        assert plan.files[1].dependencies == ["main.py"]
        assert not plan.approved

    def test_plan_serialization(self):
        plan = GenerationPlan(
            project_name="test",
            description="test",
            files=[PlannedFile(path="a.py", description="file a")],
        )
        json_str = plan.model_dump_json()
        restored = GenerationPlan.model_validate_json(json_str)
        assert restored.project_name == plan.project_name
        assert len(restored.files) == 1


class TestFileChunk:
    """FileChunkモデルのテスト。"""

    def test_full_chunk(self):
        chunk = FileChunk(
            path="src/main.py",
            content="print('hello')",
            strategy=ChunkStrategy.FULL,
            size=15,
            mtime=1700000000.0,
        )
        assert chunk.strategy == ChunkStrategy.FULL
        assert chunk.summary is None

    def test_hybrid_chunk(self):
        chunk = FileChunk(
            path="large_file.py",
            content="head...\n...\ntail...",
            strategy=ChunkStrategy.HYBRID,
            size=10000,
            mtime=1700000000.0,
            summary="大きなファイル",
        )
        assert chunk.strategy == ChunkStrategy.HYBRID
        assert chunk.summary == "大きなファイル"


class TestProjectConfig:
    """ProjectConfigモデルのテスト。"""

    def test_defaults(self):
        config = ProjectConfig()
        assert config.model == ""
        assert config.token_limit == 12000
        assert config.mode == ProjectMode.GENERATE

    def test_serialization_roundtrip(self):
        config = ProjectConfig(
            project_name="test",
            model="codellama",
            token_limit=4000,
        )
        json_str = config.model_dump_json()
        restored = ProjectConfig.model_validate_json(json_str)
        assert restored.model == "codellama"
        assert restored.token_limit == 4000


class TestMessage:
    """Messageモデルのテスト。"""

    def test_user_message(self):
        msg = Message(role="user", content="こんにちは")
        assert msg.role == "user"
        assert msg.content == "こんにちは"

    def test_assistant_message(self):
        msg = Message(role="assistant", content="はい、どうぞ")
        assert msg.role == "assistant"


class TestExceptions:
    """カスタム例外クラスのテスト。"""

    def test_inheritance(self):
        assert issubclass(OllamaConnectionError, LocalForgeError)
        assert issubclass(OllamaModelNotFoundError, LocalForgeError)
        assert issubclass(PlanParseError, LocalForgeError)
        assert issubclass(FileWriteError, LocalForgeError)
        assert issubclass(GitOperationError, LocalForgeError)
        assert issubclass(IndexBuildError, LocalForgeError)
        assert issubclass(TokenBudgetExceededWarning, LocalForgeError)
        assert issubclass(ResumeStateCorruptError, LocalForgeError)

    def test_raise_and_catch(self):
        with pytest.raises(LocalForgeError):
            raise OllamaConnectionError("接続失敗")

    def test_message(self):
        err = PlanParseError("JSONが無効です")
        assert "JSONが無効です" in str(err)


class TestGenerationLogEntry:
    """GenerationLogEntryモデルのテスト。"""

    def test_defaults(self):
        entry = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="plan",
        )
        assert entry.status == "pending"
        assert entry.prompt_tokens_estimated == 0

    def test_serialization(self):
        entry = GenerationLogEntry(
            mode="explain",
            model="codellama",
            operation="summary",
            file_path="src/main.py",
            status="completed",
        )
        json_str = entry.model_dump_json()
        restored = GenerationLogEntry.model_validate_json(json_str)
        assert restored.file_path == "src/main.py"
        assert restored.status == "completed"
