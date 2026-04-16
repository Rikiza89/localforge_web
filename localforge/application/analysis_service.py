"""
分析サービス — インデックス構築・ハイブリッドファイル読み込みの責務を担う。
ヒューリスティックサマリー（LLM不要な自明なファイル）と
LLMバッチ処理（複数ファイルを1回のLLM呼び出しで処理）を組み合わせて
インデックス構築を高速化する。
"""

from __future__ import annotations

import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

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
# 設定・データファイルの拡張子（LLMサマリー不要）
_HEURISTIC_EXTENSIONS = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".txt", ".rst", ".lock", ".env",
    ".gitignore", ".dockerignore", ".gitattributes", ".editorconfig",
})
# ロックファイル名（依存関係のロックファイル）
_LOCK_FILE_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock", "go.sum",
})
# この実質行数以下のファイルはLLMを使わない
_HEURISTIC_LINE_THRESHOLD = 8
# 1回のLLM呼び出しで処理するファイル数
_BATCH_SIZE = 5
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


def _smart_summary_python(source: str) -> Optional[str]:
    """
    Python AST を使ってLLM不要のサマリーを生成する。
    モジュールdocstring → クラス定義（+docstring） → 公開関数（+docstring）の順に抽出。
    何も抽出できなかった場合はNoneを返す。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    parts: List[str] = []

    # モジュールdocstring（最も情報量が多い）
    module_doc = ast.get_docstring(tree)
    if module_doc:
        first = module_doc.strip().split("\n")[0].strip()
        if first:
            parts.append(first)

    # トップレベルのクラス
    class_items: List[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                desc = doc.strip().split("\n")[0][:50]
                class_items.append(f"{node.name}（{desc}）")
            else:
                class_items.append(node.name)
    if class_items:
        parts.append("クラス: " + "、".join(class_items[:4]))

    # トップレベルの公開関数
    func_items: List[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                doc = ast.get_docstring(node)
                if doc:
                    desc = doc.strip().split("\n")[0][:50]
                    func_items.append(f"{node.name}（{desc}）")
                else:
                    func_items.append(node.name)
    if func_items:
        parts.append("関数: " + "、".join(func_items[:5]))

    return " | ".join(parts) if parts else None


# JS/TS エクスポート抽出パターン
_JS_NAMED_EXPORT_RE = re.compile(
    r"export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)"
)
_JS_DEFAULT_EXPORT_RE = re.compile(r"export\s+default\s+(?:class|function)?\s*(\w+)")


def _smart_summary_js_ts(path: Path, source: str) -> Optional[str]:
    """
    JS/TS ソースの先頭行からエクスポート情報を正規表現で抽出してサマリーを生成する。
    LLM不要。何も抽出できなかった場合はNoneを返す。
    """
    head = "\n".join(source.splitlines()[:100])

    named = list(dict.fromkeys(_JS_NAMED_EXPORT_RE.findall(head)))
    default_match = _JS_DEFAULT_EXPORT_RE.search(head)

    items: List[str] = []
    if default_match and default_match.group(1):
        items.append(f"デフォルト: {default_match.group(1)}")
    if named:
        items.append("エクスポート: " + "、".join(named[:6]))

    if not items:
        return None

    lang = _detect_language(path)
    return f"{lang}ファイル — " + " | ".join(items)


def _try_heuristic_summary(chunk: FileChunk) -> Optional[str]:
    """
    LLMを呼ばずにサマリーを生成できる場合はその文字列を返す。
    生成できない場合はNoneを返し、呼び出し元がLLMバッチにフォールバックする。

    優先順位:
      1. 空ファイル / ロックファイル / 設定拡張子 / 極小ファイル  → 即時テキスト
      2. Python → AST（モジュールdocstring + クラス/関数シグネチャ）
      3. JS/TS  → 正規表現（エクスポート名）
      4. それ以外 → None（LLMバッチへ）
    """
    path = Path(chunk.path)
    non_empty = [l for l in chunk.content.splitlines() if l.strip()]

    # 空ファイル
    if not non_empty:
        return "空のファイル"

    # ロックファイル
    if path.name in _LOCK_FILE_NAMES:
        return "依存関係のロックファイル"

    # 設定・データファイル（拡張子で判定）
    if path.suffix.lower() in _HEURISTIC_EXTENSIONS:
        lang = _detect_language(path)
        return f"{lang} 設定・データファイル（{len(non_empty)} 行）"

    # 極小ファイル（実質行数がしきい値以下）
    if len(non_empty) <= _HEURISTIC_LINE_THRESHOLD:
        lang = _detect_language(path)
        return f"小さな {lang} ファイル（{len(non_empty)} 行）"

    ext = path.suffix.lower()

    # Python: AST解析でdocstring + クラス/関数名を抽出
    if ext == ".py":
        return _smart_summary_python(chunk.content)  # None → LLMフォールバック

    # JS/TS: エクスポート名をパターンマッチで抽出
    if ext in (".js", ".ts", ".jsx", ".tsx", ".vue"):
        return _smart_summary_js_ts(path, chunk.content)  # None → LLMフォールバック

    # その他の言語 → LLMバッチ
    return None


def _parse_batch_summaries(response: str) -> Dict[str, str]:
    """
    バッチLLM応答から FILE: / SUMMARY: ペアを解析してパス→サマリーの辞書を返す。
    認識できない行は無視するため、LLMが余計なテキストを出力しても壊れない。
    """
    result: Dict[str, str] = {}
    current_path: Optional[str] = None
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if line.startswith("FILE:"):
            current_path = line[len("FILE:"):].strip()
        elif line.startswith("SUMMARY:") and current_path is not None:
            result[current_path] = line[len("SUMMARY:"):].strip()
            current_path = None
    return result


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
        all_chunks: List[FileChunk] = list(cached_chunks)

        if files_to_process:
            # フェーズ1: ヒューリスティックサマリー（LLM不要なファイルを即時処理）
            needs_llm: List[Tuple[Path, FileChunk]] = []
            for f, chunk in files_to_process:
                heuristic = _try_heuristic_summary(chunk)
                if heuristic is not None:
                    chunk.summary = heuristic
                    chunk.indexed_at = datetime.utcnow()
                    all_chunks.append(chunk)
                    done_count += 1
                    yield {
                        "progress": {
                            "done": done_count,
                            "total": total,
                            "current_file": chunk.path,
                        }
                    }
                else:
                    needs_llm.append((f, chunk))

            # フェーズ2: LLMバッチ処理（_BATCH_SIZE件ずつまとめて1回のLLM呼び出しで処理）
            for batch_start in range(0, len(needs_llm), _BATCH_SIZE):
                batch = needs_llm[batch_start: batch_start + _BATCH_SIZE]
                batch_chunks = [chunk for _, chunk in batch]

                try:
                    prompt = self._context.build_batch_file_summary_prompt(batch_chunks)
                    response = self._llm.generate_sync(model, prompt)
                    summaries = _parse_batch_summaries(response)
                except Exception as exc:
                    logger.warning("バッチサマリー生成エラー: %s", exc)
                    summaries = {}

                for _, chunk in batch:
                    summary = summaries.get(chunk.path, "").strip()
                    chunk.summary = summary if summary else f"サマリー生成に失敗しました"
                    chunk.indexed_at = datetime.utcnow()
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
