"""
VectorAdapter — ChromaDB + Ollama nomic-embed-text によるセマンティック検索アダプター。
.localforge/chroma/ に永続化する組み込み ChromaDB コレクションを管理する。

埋め込みは Ollama の /api/embeddings エンドポイント（nomic-embed-text:latest）を使用する。
Ollama が起動していない、またはモデルが未インストールの場合は BM25 にフォールバックする。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import requests

from localforge.domain.models import FileChunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = "nomic-embed-text:latest"
_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
_EMBED_TIMEOUT = 30  # 秒
_COLLECTION_NAME = "localforge_nomic"
_CHROMA_DIR = "chroma"


def _ollama_embed(text: str) -> Optional[List[float]]:
    """
    Ollama の /api/embeddings を呼び出して埋め込みベクトルを返す。
    Ollama 未起動またはモデル未インストール時は None を返す（警告のみ）。
    """
    try:
        resp = requests.post(
            _EMBED_URL,
            json={"model": _EMBED_MODEL, "prompt": text},
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            logger.warning("Ollama 埋め込みレスポンスに embedding フィールドがありません")
            return None
        return embedding
    except requests.exceptions.ConnectionError:
        logger.warning(
            "Ollama に接続できません — 埋め込みをスキップします。"
            " RAG を有効にするには Ollama を起動して nomic-embed-text:latest を pull してください。"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        logger.warning("Ollama 埋め込み HTTP エラー: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Ollama 埋め込みエラー: %s", exc)
        return None


class VectorAdapter:
    """
    ChromaDB を使ったベクトルインデックスの永続化・検索を担うアダプター。
    Ollama の nomic-embed-text:latest で埋め込みを生成する。
    """

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._chroma_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------

    def init_collection(self, project_root: Path) -> None:
        """
        プロジェクトルートに対応する ChromaDB コレクションを初期化する。
        .localforge/chroma/ に永続化される。
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
        logger.info("ChromaDB コレクション初期化完了: %s", chroma_path)

    def is_initialized(self) -> bool:
        return self._collection is not None

    def collection_exists(self, project_root: Path) -> bool:
        return (project_root / ".localforge" / _CHROMA_DIR).exists()

    # ------------------------------------------------------------------
    # 埋め込み生成
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        return _ollama_embed(text)

    def _chunk_to_embed_text(self, chunk: FileChunk) -> str:
        parts = [chunk.path]
        if chunk.summary:
            parts.append(chunk.summary)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # インデックス操作
    # ------------------------------------------------------------------

    def upsert_chunk(self, chunk: FileChunk) -> bool:
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
            logger.warning("ChromaDB upsert エラー: %s — %s", chunk.path, exc)
            return False

    def needs_reembedding(self, chunk: FileChunk) -> bool:
        if not self._collection:
            return True
        try:
            result = self._collection.get(ids=[chunk.path], include=["metadatas"])
            if not result["ids"]:
                return True
            meta = result["metadatas"][0]
            stored_mtime = meta.get("mtime", 0.0)
            stored_size = meta.get("size", -1)
            return abs(stored_mtime - chunk.mtime) >= 0.001 or stored_size != chunk.size
        except Exception as exc:
            logger.warning("ChromaDB get エラー: %s — %s", chunk.path, exc)
            return True

    def migrate_from_chunks(self, chunks: List[FileChunk]) -> int:
        embedded = 0
        for chunk in chunks:
            if not chunk.summary:
                continue
            if self.needs_reembedding(chunk):
                if self.upsert_chunk(chunk):
                    embedded += 1
        logger.info("ChromaDB 移行完了: %d 件を埋め込みました", embedded)
        return embedded

    def delete_chunk(self, path: str) -> None:
        if not self._collection:
            return
        try:
            self._collection.delete(ids=[path])
        except Exception as exc:
            logger.warning("ChromaDB delete エラー: %s — %s", path, exc)

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
        クエリに意味的に近い FileChunk を返す。
        ChromaDB が初期化されていない、または Ollama が利用不可の場合は BM25 にフォールバックする。
        """
        if not self._collection:
            return _bm25_fallback(all_chunks, query, top_n)

        try:
            count = self._collection.count()
        except Exception:
            count = 0

        if count == 0:
            return _bm25_fallback(all_chunks, query, top_n)

        embedding = self._embed(query)
        if embedding is None:
            return _bm25_fallback(all_chunks, query, top_n)

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_n, count),
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("ChromaDB クエリエラー: %s", exc)
            return _bm25_fallback(all_chunks, query, top_n)

        chunk_map = {c.path: c for c in all_chunks}
        found: List[FileChunk] = []
        for meta in results.get("metadatas", [[]])[0]:
            path = meta.get("path", "")
            if path in chunk_map:
                found.append(chunk_map[path])

        return found if found else _bm25_fallback(all_chunks, query, top_n)


def _bm25_fallback(chunks: List[FileChunk], query: str, top_n: int) -> List[FileChunk]:
    from localforge.infrastructure.bm25_adapter import get_top_chunks_bm25
    return get_top_chunks_bm25(chunks, query, top_n)
