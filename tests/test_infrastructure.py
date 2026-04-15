"""
インフラストラクチャ層の単体テスト — filesystem, index, git アダプターのテスト。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pytest

from localforge.domain.models import (
    ChunkStrategy,
    FileChunk,
    GenerationLogEntry,
    ProjectIndex,
)
from localforge.infrastructure.filesystem_adapter import (
    CODE_EXTENSIONS,
    FileSystemAdapter,
)
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.domain.exceptions import FileWriteError, GitOperationError


class TestFileSystemAdapter:
    """FileSystemAdapterの単体テスト。"""

    def test_write_and_read_text(self, tmp_path: Path):
        fs = FileSystemAdapter()
        file_path = tmp_path / "test.txt"
        content = "Hello, LocalForge!\n日本語テスト"

        fs.write_text(file_path, content)
        result = fs.read_text(file_path)
        assert result == content

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        fs = FileSystemAdapter()
        nested_path = tmp_path / "a" / "b" / "c" / "file.py"
        fs.write_text(nested_path, "# content")
        assert nested_path.exists()

    def test_list_files_with_extensions(self, tmp_path: Path):
        fs = FileSystemAdapter()
        (tmp_path / "main.py").write_text("# py", encoding="utf-8")
        (tmp_path / "index.html").write_text("<html>", encoding="utf-8")
        (tmp_path / "README.md").write_text("# md", encoding="utf-8")
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01")

        py_files = fs.list_files(tmp_path, extensions=[".py"])
        assert len(py_files) == 1
        assert py_files[0].name == "main.py"

    def test_list_files_excludes_ignored_dirs(self, tmp_path: Path):
        fs = FileSystemAdapter()
        (tmp_path / "main.py").write_text("# main", encoding="utf-8")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "lib.js").write_text("// lib", encoding="utf-8")

        files = fs.list_files(tmp_path, extensions=[".py", ".js"])
        names = [f.name for f in files]
        assert "main.py" in names
        assert "lib.js" not in names

    def test_has_code_files_true(self, tmp_path: Path):
        fs = FileSystemAdapter()
        (tmp_path / "main.py").write_text("# code", encoding="utf-8")
        assert fs.has_code_files(tmp_path)

    def test_has_code_files_false(self, tmp_path: Path):
        fs = FileSystemAdapter()
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        assert not fs.has_code_files(tmp_path)

    def test_build_file_tree(self, tmp_path: Path):
        fs = FileSystemAdapter()
        (tmp_path / "main.py").write_text("# main", encoding="utf-8")
        src = tmp_path / "src"
        src.mkdir()
        (src / "utils.py").write_text("# utils", encoding="utf-8")

        tree = fs.build_file_tree(tmp_path)
        names = {n.name for n in tree}
        assert "main.py" in names
        assert "src" in names

    def test_get_mtime_size(self, tmp_path: Path):
        fs = FileSystemAdapter()
        file_path = tmp_path / "file.txt"
        file_path.write_text("test content", encoding="utf-8")

        mtime, size = fs.get_mtime_size(file_path)
        assert mtime > 0
        assert size > 0

    def test_exists(self, tmp_path: Path):
        fs = FileSystemAdapter()
        existing = tmp_path / "exists.py"
        existing.write_text("", encoding="utf-8")

        assert fs.exists(existing)
        assert not fs.exists(tmp_path / "nonexistent.py")

    def test_code_extensions_set(self):
        assert ".py" in CODE_EXTENSIONS
        assert ".js" in CODE_EXTENSIONS
        assert ".ts" in CODE_EXTENSIONS
        assert ".bin" not in CODE_EXTENSIONS


class TestIndexAdapter:
    """IndexAdapterの単体テスト。"""

    def test_save_and_load_chunks(self, tmp_path: Path):
        adapter = IndexAdapter()
        index_path = tmp_path / "index.jsonl"

        chunks = [
            FileChunk(
                path="main.py",
                content="def main(): pass",
                strategy=ChunkStrategy.FULL,
                size=17,
                mtime=1700000000.0,
                summary="メイン関数",
            ),
            FileChunk(
                path="utils.py",
                content="def helper(): pass",
                strategy=ChunkStrategy.HYBRID,
                size=18,
                mtime=1700000001.0,
                summary="ユーティリティ",
            ),
        ]

        adapter.save_chunks(index_path, chunks)
        loaded = adapter.load_chunks(index_path)

        assert len(loaded) == 2
        assert loaded[0].path == "main.py"
        assert loaded[0].summary == "メイン関数"
        assert loaded[1].strategy == ChunkStrategy.HYBRID

    def test_load_chunks_nonexistent_file(self, tmp_path: Path):
        adapter = IndexAdapter()
        result = adapter.load_chunks(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_save_and_load_index(self, tmp_path: Path):
        adapter = IndexAdapter()
        index_path = tmp_path / "project_index.json"

        index = ProjectIndex(
            project_root=str(tmp_path),
            project_name="test_proj",
            summary="テストプロジェクトの概要",
            total_files=5,
            indexed_files=5,
        )

        adapter.save_index(index_path, index)
        loaded = adapter.load_index(index_path)

        assert loaded is not None
        assert loaded.project_name == "test_proj"
        assert loaded.summary == "テストプロジェクトの概要"
        assert loaded.total_files == 5

    def test_load_index_nonexistent(self, tmp_path: Path):
        adapter = IndexAdapter()
        result = adapter.load_index(tmp_path / "nonexistent.json")
        assert result is None

    def test_append_and_load_log_entries(self, tmp_path: Path):
        adapter = IndexAdapter()
        log_path = tmp_path / "log.jsonl"

        entry1 = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="plan",
            status="completed",
        )
        entry2 = GenerationLogEntry(
            mode="generate",
            model="llama3.2",
            operation="generate_file",
            file_path="main.py",
            status="pending",
        )

        adapter.append_log_entry(log_path, entry1)
        adapter.append_log_entry(log_path, entry2)

        loaded = adapter.load_log_entries(log_path)
        assert len(loaded) == 2
        assert loaded[0].operation == "plan"
        assert loaded[1].file_path == "main.py"

    def test_load_log_entries_empty(self, tmp_path: Path):
        adapter = IndexAdapter()
        result = adapter.load_log_entries(tmp_path / "nonexistent.jsonl")
        assert result == []


class TestGitAdapter:
    """GitAdapterの単体テスト。"""

    def test_is_git_repo_false(self, tmp_path: Path):
        git = GitAdapter()
        assert not git._is_git_repo(tmp_path)

    def test_init_creates_git_dir(self, tmp_path: Path):
        git = GitAdapter()
        git.init(tmp_path)
        assert (tmp_path / ".git").is_dir()

    def test_init_creates_gitignore(self, tmp_path: Path):
        git = GitAdapter()
        git.init(tmp_path)
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()

    def test_get_log_non_repo(self, tmp_path: Path):
        git = GitAdapter()
        result = git.get_log(tmp_path)
        assert result == []

    def test_get_status_non_repo(self, tmp_path: Path):
        git = GitAdapter()
        result = git.get_status(tmp_path)
        assert result == ""

    def test_get_current_branch_non_repo(self, tmp_path: Path):
        git = GitAdapter()
        result = git.get_current_branch(tmp_path)
        assert result == ""

    def test_commit_all(self, tmp_path: Path):
        git = GitAdapter()
        git.init(tmp_path)

        (tmp_path / "test.txt").write_text("hello", encoding="utf-8")
        commit_hash = git.commit_all(tmp_path, "テストコミット")

        # コミットが成功すればハッシュが返る（空でも可）
        assert isinstance(commit_hash, str)

    def test_get_log_after_commit(self, tmp_path: Path):
        git = GitAdapter()
        git.init(tmp_path)
        (tmp_path / "file.py").write_text("# test", encoding="utf-8")
        git.commit_all(tmp_path, "初回コミット")

        entries = git.get_log(tmp_path, max_entries=5)
        assert len(entries) >= 1
        assert entries[0]["message"] == "初回コミット"

    def test_get_diff_empty(self, tmp_path: Path):
        git = GitAdapter()
        git.init(tmp_path)
        result = git.get_diff(tmp_path)
        assert isinstance(result, str)
