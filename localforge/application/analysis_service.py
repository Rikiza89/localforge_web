"""
分析サービス — インデックス構築・ハイブリッドファイル読み込みの責務を担う。
parallel file summarization に ThreadPoolExecutor(max_workers=3) を使用する。
"""

from __future__ import annotations

import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

from localforge.application.context_service import ContextService
from localforge.domain.exceptions import IndexBuildError
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
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ハイブリッド戦略のしきい値（行数）
_HYBRID_THRESHOLD = 200
# ハイブリッド戦略: 先頭の行数
_HYBRID_HEAD = 80
# ハイブリッド戦略: 末尾の行数
_HYBRID_TAIL = 40
# 並列サマリー生成のワーカー数
_MAX_WORKERS = 3
# .localforgeディレクトリ名
_LOCALFORGE_DIR = ".localforge"


def _detect_language(path: Path) -> str:
    """
    ファイルの拡張子から言語を推定する。

    Args:
        path: ファイルパス

    Returns:
        言語名文字列
    """
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".go": "go",
        ".rs": "rust", ".java": "java", ".html": "html",
        ".css": "css", ".scss": "css", ".vue": "vue",
        ".rb": "ruby", ".php": "php", ".cs": "csharp",
        ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".json": "json", ".sql": "sql",
        ".md": "markdown",
    }
    return ext_map.get(path.suffix.lower(), "text")


def _extract_python_landmarks(source: str) -> List[str]:
    """
    ASTを使用してPythonソースコードの構造的ランドマーク（クラス・関数・インポート）を抽出する。
    正規表現は使用しない。

    Args:
        source: Pythonソースコード文字列

    Returns:
        ランドマーク行のリスト
    """
    landmarks: List[str] = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                line_num = node.lineno
                src_lines = source.splitlines()
                if 0 < line_num <= len(src_lines):
                    landmarks.append(src_lines[line_num - 1])
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                line_num = node.lineno
                src_lines = source.splitlines()
                if 0 < line_num <= len(src_lines):
                    landmarks.append(f"# L{line_num}: " + src_lines[line_num - 1])
    except SyntaxError:
        logger.debug("PythonランドマークASTエラー（構文エラー）")
    return landmarks


def _extract_js_ts_landmarks(source: str) -> List[str]:
    """
    JavaScript/TypeScriptソースコードの構造的ランドマークを行パターンで抽出する。
    （ASTは使用せず、export/function/class/import 行を収集）

    Args:
        source: JS/TSソースコード文字列

    Returns:
        ランドマーク行のリスト
    """
    landmarks: List[str] = []
    keywords = ("export ", "import ", "function ", "class ", "const ", "async ")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if any(stripped.startswith(kw) for kw in keywords):
            landmarks.append(f"# L{i}: {line}")
    return landmarks


class AnalysisService:
    """
    コードベース分析のためのインデックス構築を担当するサービスクラス。
    ファイルごとのハイブリッド読み込み・並列サマリー生成・増分再インデックスをサポートする。
    """

    def __init__(
        self,
        fs: FileSystemAdapter,
        index_adapter: IndexAdapter,
        llm: OllamaClient,
        context: ContextService,
    ) -> None:
        """
        AnalysisServiceを初期化する。

        Args:
            fs: ファイルシステムアダプター
            index_adapter: インデックスアダプター
            llm: OllamaクライアントLLMバックエンド
            context: コンテキストサービス
        """
        self._fs = fs
        self._index_adapter = index_adapter
        self._llm = llm
        self._context = context

    def read_file_chunk(self, path: Path, root: Path) -> FileChunk:
        """
        ハイブリッド読み込み戦略でファイルを読み込みFileChunkを生成する。
        - ≤200行: フル読み込み (strategy=full)
        - >200行: 先頭80行 + 末尾40行 + 構造的ランドマーク (strategy=hybrid)

        Args:
            path: ファイルの絶対パス
            root: プロジェクトルート（相対パス計算用）

        Returns:
            FileChunk
        """
        mtime, size = self._fs.get_mtime_size(path)
        try:
            source = self._fs.read_text(path)
        except Exception as exc:
            logger.warning("ファイル読み込みエラー: %s — %s", path, exc)
            source = ""

        lines = source.splitlines()
        lang = _detect_language(path)
        rel_path = str(path.relative_to(root))

        if len(lines) <= _HYBRID_THRESHOLD:
            content = source
            strategy = ChunkStrategy.FULL
        else:
            head = lines[:_HYBRID_HEAD]
            tail = lines[-_HYBRID_TAIL:]
            landmarks: List[str] = []

            if lang == "python":
                landmarks = _extract_python_landmarks(source)
            elif lang in ("javascript", "typescript"):
                landmarks = _extract_js_ts_landmarks(source)

            # 重複除去してランドマークを結合
            seen = set()
            unique_landmarks: List[str] = []
            for lm in landmarks:
                if lm not in seen:
                    seen.add(lm)
                    unique_landmarks.append(lm)

            sections = [
                "\n".join(head),
                "... (省略) ...",
                "\n".join(tail),
            ]
            if unique_landmarks:
                sections.insert(2, "--- 構造的ランドマーク ---\n" + "\n".join(unique_landmarks))

            content = "\n".join(sections)
            strategy = ChunkStrategy.HYBRID

        return FileChunk(
            path=rel_path,
            content=content,
            strategy=strategy,
            size=size,
            mtime=mtime,
            language=lang,
        )

    def build_index(
        self,
        root: Path,
        model: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Generator[dict, None, None]:
        """
        コードベース全体のインデックスを構築し、進捗をSSEイベントとして生成する。
        増分再インデックスをサポート（mtime+sizeが変化したファイルのみ再処理）。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            on_progress: 進捗コールバック（done, total, current_file）

        Yields:
            SSEペイロード辞書
        """
        lf_dir = root / _LOCALFORGE_DIR
        index_path = lf_dir / "index.jsonl"

        # 既存チャンクを読み込んでmtime+sizeでキャッシュを作成
        existing_chunks = self._index_adapter.load_chunks(index_path)
        chunk_cache: dict[str, FileChunk] = {
            c.path: c for c in existing_chunks
        }

        # コードファイル一覧を取得
        code_files = self._fs.list_code_files(root)
        # .localforgeディレクトリのファイルを除外
        code_files = [
            f for f in code_files
            if _LOCALFORGE_DIR not in f.parts
        ]
        total = len(code_files)

        if total == 0:
            yield {"progress": {"done": 0, "total": 0, "current_file": ""}}
            return

        yield {"progress": {"done": 0, "total": total, "current_file": ""}}

        # 変更ファイルの特定（増分インデックス）
        files_to_process: List[Tuple[Path, FileChunk]] = []
        cached_chunks: List[FileChunk] = []

        for f in code_files:
            rel = str(f.relative_to(root))
            mtime, size = self._fs.get_mtime_size(f)
            cached = chunk_cache.get(rel)

            if cached and abs(cached.mtime - mtime) < 0.001 and cached.size == size and cached.summary:
                # 変更なし → キャッシュを使用
                cached_chunks.append(cached)
            else:
                chunk = self.read_file_chunk(f, root)
                files_to_process.append((f, chunk))

        done_count = len(cached_chunks)

        # 並列サマリー生成
        all_chunks: List[FileChunk] = list(cached_chunks)

        if files_to_process:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                future_map = {}
                for _, chunk in files_to_process:
                    prompt = self._context.build_file_summary_prompt(
                        file_path=chunk.path,
                        content=chunk.content,
                        extension=Path(chunk.path).suffix,
                    )
                    future = executor.submit(self._llm.generate_sync, model, prompt)
                    future_map[future] = chunk

                for future in as_completed(future_map):
                    chunk = future_map[future]
                    try:
                        summary = future.result()
                        chunk.summary = summary.strip()
                        chunk.indexed_at = datetime.utcnow()
                    except Exception as exc:
                        logger.warning("サマリー生成エラー: %s — %s", chunk.path, exc)
                        chunk.summary = f"サマリー生成に失敗しました: {exc}"

                    all_chunks.append(chunk)
                    done_count += 1
                    yield {
                        "progress": {
                            "done": done_count,
                            "total": total,
                            "current_file": chunk.path,
                        }
                    }

        # インデックスを保存
        try:
            self._index_adapter.save_chunks(index_path, all_chunks)
        except Exception as exc:
            raise IndexBuildError(f"インデックス保存失敗: {exc}") from exc

        # ProjectIndexを構築して保存
        project_index = self._build_project_index(root, model, all_chunks)
        pi_path = lf_dir / "project_index.json"
        self._index_adapter.save_index(pi_path, project_index)

        yield {"progress": {"done": total, "total": total, "current_file": "完了"}}
        yield {"done": True}

    def _build_project_index(
        self,
        root: Path,
        model: str,
        chunks: List[FileChunk],
    ) -> ProjectIndex:
        """
        ファイルチャンクからProjectIndexを構築する内部メソッド。

        Args:
            root: プロジェクトルート
            model: 使用するモデル名
            chunks: FileChunkのリスト

        Returns:
            ProjectIndex
        """
        # ファイルツリーのテキスト表現
        folder_tree = self._build_tree_text(root)

        # ルート設定ファイルの内容
        root_config_content = self._read_root_configs(root)

        # サマリーリスト
        file_summaries = [
            (c.path, c.summary or "") for c in chunks if c.summary
        ]

        # ProjectIndex概要をLLMで生成
        prompt = self._context.build_project_index_prompt(
            file_summaries=file_summaries,
            folder_tree=folder_tree,
            root_configs=root_config_content,
        )
        try:
            summary = self._llm.generate_sync(model, prompt).strip()
        except Exception as exc:
            logger.error("ProjectIndex概要生成エラー: %s", exc)
            summary = f"プロジェクト概要の生成に失敗しました: {exc}"

        return ProjectIndex(
            project_root=str(root),
            project_name=root.name,
            summary=summary,
            file_chunks=chunks,
            total_files=len(chunks),
            indexed_files=len(chunks),
            updated_at=datetime.utcnow(),
        )

    def _build_tree_text(self, root: Path, prefix: str = "", path: Optional[Path] = None) -> str:
        """
        ディレクトリツリーをUnicodeボックス文字で表現する内部メソッド。

        Args:
            root: ルートディレクトリ
            prefix: インデントプレフィックス
            path: 現在のパス（省略時はroot）

        Returns:
            ツリーテキスト
        """
        if path is None:
            path = root

        skip_dirs = {".git", ".localforge", "__pycache__", "node_modules", ".venv", "venv"}
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return ""

        lines = []
        filtered = [e for e in entries if not (e.is_dir() and e.name in skip_dirs)]

        for i, entry in enumerate(filtered):
            is_last = i == len(filtered) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                subtree = self._build_tree_text(root, prefix + extension, entry)
                if subtree:
                    lines.append(subtree)

        return "\n".join(lines)

    def _read_root_configs(self, root: Path) -> str:
        """
        ルートディレクトリの設定ファイルを読み込む内部メソッド。

        Args:
            root: プロジェクトルート

        Returns:
            設定ファイルの内容文字列（複数ファイルを結合）
        """
        config_names = [
            "package.json", "pyproject.toml", "setup.py", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "Makefile",
            "docker-compose.yml", "docker-compose.yaml",
        ]
        parts = []
        for name in config_names:
            config_path = root / name
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"--- {name} ---\n{content[:500]}")
                except OSError:
                    pass
        return "\n\n".join(parts)

    def load_project_index(self, root: Path) -> Optional[ProjectIndex]:
        """
        保存済みのProjectIndexを読み込む。

        Args:
            root: プロジェクトルート

        Returns:
            ProjectIndex（存在しない場合はNone）
        """
        pi_path = root / _LOCALFORGE_DIR / "project_index.json"
        return self._index_adapter.load_index(pi_path)

    def get_top_chunks_by_keywords(
        self,
        chunks: List[FileChunk],
        query: str,
        top_n: int = 5,
    ) -> List[FileChunk]:
        """
        キーワードオーバーラップでファイルチャンクをランキングして上位N件を返す。

        Args:
            chunks: 全FileChunkのリスト
            query: 検索クエリ文字列
            top_n: 返す件数

        Returns:
            上位N件のFileChunkリスト
        """
        query_words = set(query.lower().split())

        def score(chunk: FileChunk) -> int:
            text = f"{chunk.path} {chunk.summary or ''} {chunk.content[:200]}".lower()
            return sum(1 for w in query_words if w in text)

        sorted_chunks = sorted(chunks, key=score, reverse=True)
        return sorted_chunks[:top_n]
