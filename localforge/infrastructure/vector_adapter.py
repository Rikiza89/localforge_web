"""
VectorAdapter — ChromaDB + Ollama埋め込みを使ったセマンティック検索アダプター。
.localforge/chroma/ に永続化する組み込みChromaDBコレクションを管理する。
増分更新をサポート: mtime+sizeが変化したファイルのみ再埋め込みする。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import requests

from localforge.domain.models import FileChunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = "nomic-embed-text:latest"
_COLLECTION_NAME = "localforge_index"
_CHROMA_DIR = "chroma"
_CONNECT_TIMEOUT = 5
_EMBED_TIMEOUT = 60


class VectorAdapter:
    """
    ChromaDBを使ったベクトルインデックスの永続化・検索を担うアダプター。
    Ollamaの /api/embeddings エンドポイントで埋め込みを生成する。
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._client = None
        self._collection = None
        self._chroma_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------

    def init_collection(self, project_root: Path) -> None:
        """
        プロジェクトルートに対応するChromaDBコレクションを初期化する。
        .localforge/chroma/ に永続化される。

        Args:
            project_root: プロジェクトのルートディレクトリ
        """
        import chromadb
        from chromadb.config import Settings

        chroma_path = project_root / ".localforge" / _CHROMA_DIR
        chroma_path.mkdir(parents=True, exist_ok=True)

        if self._chroma_path == chroma_path and self._collection is not None:
            return

        self._chroma_path = chroma_path
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDBコレクション初期化完了: %s", chroma_path)

    def is_initialized(self) -> bool:
        return self._collection is not None

    def collection_exists(self, project_root: Path) -> bool:
        """ChromaDBコレクションがすでにディスク上に存在するかを確認する。"""
        return (project_root / ".localforge" / _CHROMA_DIR).exists()

    # ------------------------------------------------------------------
    # 埋め込み生成
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Ollama /api/embeddings エンドポイントで埋め込みベクトルを生成する。

        Args:
            text: 埋め込む文字列

        Returns:
            埋め込みベクトル（失敗時はNone）
        """
        try:
            resp = self._session.post(
                f"{self._base_url}/api/embeddings",
                json={"model": _EMBED_MODEL, "prompt": text},
                timeout=(_CONNECT_TIMEOUT, _EMBED_TIMEOUT),
            )
            resp.raise_for_status()
            return resp.json().get("embedding")
        except Exception as exc:
            logger.warning("埋め込み生成エラー: %s", exc)
            return None

    def _chunk_to_embed_text(self, chunk: FileChunk) -> str:
        """ファイルパスとサマリーを結合した埋め込み用テキストを生成する。"""
        parts = [chunk.path]
        if chunk.summary:
            parts.append(chunk.summary)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # インデックス操作
    # ------------------------------------------------------------------

    def upsert_chunk(self, chunk: FileChunk) -> bool:
        """
        単一FileChunkをベクトルインデックスに追加または更新する。
        summaryが空の場合はスキップする。

        Args:
            chunk: 追加・更新するFileChunk

        Returns:
            成功した場合True
        """
        if not self._collection:
            logger.warning("コレクションが初期化されていません")
            return False
        if not chunk.summary:
            return False

        text = self._chunk_to_embed_text(chunk)
        embedding = self._embed(text)
        if embedding is None:
            return False

        try:
            self._collection.upsert(
                ids=[chunk.path],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "path": chunk.path,
                    "mtime": chunk.mtime,
                    "size": chunk.size,
                    "language": chunk.language or "",
                    "summary": chunk.summary or "",
                }],
            )
            return True
        except Exception as exc:
            logger.warning("ChromaDB upsertエラー: %s — %s", chunk.path, exc)
            return False

    def needs_reembedding(self, chunk: FileChunk) -> bool:
        """
        指定チャンクがChromaDBに存在しない、またはmtime/sizeが変化している場合Trueを返す。

        Args:
            chunk: チェックするFileChunk

        Returns:
            再埋め込みが必要な場合True
        """
        if not self._collection:
            return True
        try:
            result = self._collection.get(
                ids=[chunk.path],
                include=["metadatas"],
            )
            if not result["ids"]:
                return True
            meta = result["metadatas"][0]
            stored_mtime = meta.get("mtime", 0.0)
            stored_size = meta.get("size", -1)
            return abs(stored_mtime - chunk.mtime) >= 0.001 or stored_size != chunk.size
        except Exception as exc:
            logger.warning("ChromaDB getエラー: %s — %s", chunk.path, exc)
            return True

    def migrate_from_chunks(self, chunks: List[FileChunk]) -> int:
        """
        既存のJSONLインデックスからChromaDBへ一括移行する。
        summaryが存在し、まだ埋め込まれていないチャンクのみ処理する。

        Args:
            chunks: 移行するFileChunkのリスト

        Returns:
            埋め込みを生成したファイル数
        """
        embedded = 0
        for chunk in chunks:
            if not chunk.summary:
                continue
            if self.needs_reembedding(chunk):
                if self.upsert_chunk(chunk):
                    embedded += 1
        logger.info("ChromaDB移行完了: %d件を埋め込みました", embedded)
        return embedded

    def delete_chunk(self, path: str) -> None:
        """
        指定パスのチャンクをコレクションから削除する。

        Args:
            path: 削除するファイルの相対パス
        """
        if not self._collection:
            return
        try:
            self._collection.delete(ids=[path])
        except Exception as exc:
            logger.warning("ChromaDB deleteエラー: %s — %s", path, exc)

    # ------------------------------------------------------------------
    # セマンティック検索
    # ------------------------------------------------------------------

    def get_top_chunks_semantic(
        self,
        all_chunks: List[FileChunk],
        query: str,
        top_n: int = 5,
    ) -> List[FileChunk]:
        """
        クエリに意味的に近いFileChunkを返す。
        ChromaDBが初期化されていない場合はキーワード検索にフォールバックする。

        Args:
            all_chunks: 全FileChunkのリスト（フォールバック用・パス解決用）
            query: 検索クエリ
            top_n: 返す件数

        Returns:
            上位N件のFileChunkリスト
        """
        if not self._collection:
            return _keyword_fallback(all_chunks, query, top_n)

        try:
            count = self._collection.count()
        except Exception:
            count = 0

        if count == 0:
            return _keyword_fallback(all_chunks, query, top_n)

        embedding = self._embed(query)
        if embedding is None:
            return _keyword_fallback(all_chunks, query, top_n)

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_n, count),
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("ChromaDBクエリエラー: %s", exc)
            return _keyword_fallback(all_chunks, query, top_n)

        chunk_map = {c.path: c for c in all_chunks}
        found: List[FileChunk] = []
        for meta in results.get("metadatas", [[]])[0]:
            path = meta.get("path", "")
            if path in chunk_map:
                found.append(chunk_map[path])

        if not found:
            return _keyword_fallback(all_chunks, query, top_n)

        return found


def _keyword_fallback(chunks: List[FileChunk], query: str, top_n: int) -> List[FileChunk]:
    """ChromaDB未使用時のキーワードベースフォールバック検索。"""
    query_words = set(query.lower().split())

    def score(chunk: FileChunk) -> int:
        text = f"{chunk.path} {chunk.summary or ''} {chunk.content[:200]}".lower()
        return sum(1 for w in query_words if w in text)

    return sorted(chunks, key=score, reverse=True)[:top_n]
