"""
ファイルシステムアダプター — pathlib.Pathを使用したファイル読み書き・ディレクトリ操作の実装。
FileSystemPortインターフェースを実装する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from localforge.domain.exceptions import FileWriteError
from localforge.domain.models import FileNode, FileStatus

logger = logging.getLogger(__name__)

# デフォルトで除外するディレクトリ名
_DEFAULT_IGNORE_DIRS = frozenset({
    ".git", ".localforge", "__pycache__", ".venv", "venv",
    "node_modules", ".tox", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.egg-info",
})

# コードファイルとして認識する拡張子
CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".html", ".css", ".scss", ".sass", ".vue", ".svelte",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json",
    ".md", ".rst", ".txt", ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl",
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".ods", ".odp",
})


class FileSystemAdapter:
    """
    ファイルシステム操作を提供するアダプタークラス。
    すべてのパス操作にpathlib.Pathを使用する。
    """

    def read_text(self, path: Path) -> str:
        """
        テキストファイルを読み込む。

        Args:
            path: ファイルの絶対パス

        Returns:
            ファイルの内容（文字列）
        """
        return path.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: Path, content: str) -> None:
        """
        テキストファイルを書き込む（親ディレクトリは自動作成）。

        Args:
            path: ファイルの絶対パス
            content: 書き込むテキスト内容

        Raises:
            FileWriteError: 書き込みに失敗した場合
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.debug("ファイル書き込み完了: %s", path)
        except OSError as exc:
            raise FileWriteError(f"ファイル書き込みに失敗しました: {path} — {exc}") from exc

    def list_files(
        self,
        root: Path,
        extensions: Optional[Iterable[str]] = None,
        ignore_dirs: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """
        ディレクトリ以下のファイル一覧を返す。

        Args:
            root: 検索対象のルートディレクトリ
            extensions: フィルタする拡張子のリスト（省略時はすべて）
            ignore_dirs: 除外するディレクトリ名のリスト

        Returns:
            ファイルパスのリスト（ソート済み）
        """
        ext_set = frozenset(extensions) if extensions else None
        skip_dirs = _DEFAULT_IGNORE_DIRS | (
            frozenset(ignore_dirs) if ignore_dirs else frozenset()
        )

        result: List[Path] = []
        self._walk(root, root, ext_set, skip_dirs, result)
        return sorted(result)

    def _walk(
        self,
        base: Path,
        current: Path,
        ext_set: Optional[frozenset],
        skip_dirs: frozenset,
        result: List[Path],
    ) -> None:
        """再帰的にディレクトリを走査してファイルを収集する内部メソッド。"""
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            logger.warning("アクセス権限エラー: %s", current)
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name not in skip_dirs:
                    self._walk(base, entry, ext_set, skip_dirs, result)
            elif entry.is_file():
                if ext_set is None or entry.suffix.lower() in ext_set:
                    result.append(entry)

    def list_code_files(self, root: Path) -> List[Path]:
        """
        コードファイル（CODE_EXTENSIONS）のみを一覧返す。

        Args:
            root: 検索対象のルートディレクトリ

        Returns:
            コードファイルパスのリスト
        """
        return self.list_files(root, extensions=CODE_EXTENSIONS)

    def has_code_files(self, root: Path) -> bool:
        """
        ディレクトリ以下にコードファイルが存在するかどうかを確認する。

        Args:
            root: 確認するディレクトリ

        Returns:
            コードファイルが存在すればTrue
        """
        for _ in self._iter_code_files_limit(root, limit=1):
            return True
        return False

    def _iter_code_files_limit(self, root: Path, limit: int) -> Iterable[Path]:
        """指定件数だけコードファイルを列挙する内部メソッド（早期終了用）。"""
        count = 0
        for path in self.list_code_files(root):
            yield path
            count += 1
            if count >= limit:
                break

    def build_file_tree(self, root: Path) -> List[FileNode]:
        """
        ディレクトリのファイルツリーを構築する。

        Args:
            root: ツリーのルートディレクトリ

        Returns:
            FileNodeのリスト（階層構造）
        """
        return self._build_nodes(root, root)

    def _build_nodes(self, base: Path, current: Path) -> List[FileNode]:
        """再帰的にFileNodeのツリーを構築する内部メソッド。"""
        nodes: List[FileNode] = []
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return nodes

        for entry in entries:
            rel = str(entry.relative_to(base))
            if entry.is_dir():
                if entry.name.startswith(".") and entry.name not in {".localforge"}:
                    continue
                if entry.name in _DEFAULT_IGNORE_DIRS:
                    continue
                children = self._build_nodes(base, entry)
                nodes.append(FileNode(
                    name=entry.name,
                    path=rel,
                    is_dir=True,
                    children=children,
                ))
            elif entry.is_file():
                try:
                    stat = entry.stat()
                    nodes.append(FileNode(
                        name=entry.name,
                        path=rel,
                        is_dir=False,
                        size=stat.st_size,
                        modified_at=stat.st_mtime,
                    ))
                except OSError:
                    nodes.append(FileNode(name=entry.name, path=rel, is_dir=False))

        return nodes

    def get_mtime_size(self, path: Path) -> Tuple[float, int]:
        """
        ファイルの最終更新時刻とサイズを返す。

        Args:
            path: ファイルの絶対パス

        Returns:
            (mtime, size) のタプル
        """
        stat = path.stat()
        return stat.st_mtime, stat.st_size

    def exists(self, path: Path) -> bool:
        """
        パスが存在するかどうかを確認する。

        Args:
            path: 確認するパス

        Returns:
            存在すればTrue
        """
        return path.exists()

    def read_lines_range(self, path: Path, start: int, end: int) -> List[str]:
        """
        ファイルの指定行範囲を読み込む。

        Args:
            path: ファイルパス
            start: 開始行（0ベース）
            end: 終了行（0ベース、exclusive）

        Returns:
            行のリスト
        """
        lines = self.read_text(path).splitlines()
        return lines[start:end]

    def count_lines(self, path: Path) -> int:
        """
        ファイルの行数を返す。

        Args:
            path: ファイルパス

        Returns:
            行数
        """
        return len(self.read_text(path).splitlines())
