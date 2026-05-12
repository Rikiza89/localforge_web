"""
VectorAdapter — ChromaDB + sentence-transformers を使ったセマンティック検索アダプター。
.localforge/chroma/ に永続化する組み込み ChromaDB コレクションを管理する。

埋め込みモデルは sentence-transformers の all-MiniLM-L6-v2 (384 次元, ~25 MB) を
プロセス内で実行する。Ollama への HTTP 呼び出しは不要。
インクリメンタル更新をサポート: mtime+size が変化したファイルのみ再埋め込みする。

all-MiniLM-L6-v2 は初回起動時に HuggingFace Hub から自動ダウンロードされ、
~/.cache/huggingface/hub/ にキャッシュされる。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from localforge.domain.models import FileChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ST_MODEL_NAME = "all-MiniLM-L6-v2"
# nomic-embed-text (768 次元) との次元衝突を避けるため別コレクション名を使用する
_COLLECTION_NAME = "localforge_minilm"
_CHROMA_DIR = "chroma"

# モジュールレベルのモデルキャッシュ（プロセス内で一度だけロードする）
_st_model_instance = None
_st_model_load_attempted = False


def _load_st_model():
    """
    sentence-transformers モデルをロードしてキャッシュする。
    インストールされていない場合は None を返す（警告のみ、クラッシュしない）。
    """
    global _st_model_instance, _st_model_load_attempted
    if _st_model_load_attempted:
        return _st_model_instance
    _st_model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("sentence-transformers モデルをロード中: %s", _ST_MODEL_NAME)
        _st_model_instance = SentenceTransformer(_ST_MODEL_NAME)
        logger.info("sentence-transformers モデルのロード完了: %s", _ST_MODEL_NAME)
    except ImportError:
        logger.warning(
            "sentence-transformers がインストールされていません。"
            " ベクトル埋め込みは無効になり、BM25 検索のみ使用されます。"
            " 有効にするには: pip install sentence-transformers"
        )
        _st_model_instance = None
    except Exception as exc:
        logger.warning("sentence-transformers モデルのロードに失敗しました: %s", exc)
        _st_model_instance = None
    return _st_model_instance

class VectorAdapter:
    """
    ChromaDB を使ったベクトルインデックスの永続化・検索を担うアダプター。
    sentence-transformers の all-MiniLM-L6-v2 でプロセス内埋め込みを生成する。
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
        logger.info("ChromaDB コレクション初期化完了: %s", chroma_path)

    def is_initialized(self) -> bool:
        return self._collection is not None

    def collection_exists(self, project_root: Path) -> bool:
        """ChromaDB コレクションがすでにディスク上に存在するかを確認する。"""
        return (project_root / ".localforge" / _CHROMA_DIR).exists()

    # ------------------------------------------------------------------
    # 埋め込み生成（sentence-transformers プロセス内実行）
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        sentence-transformers で埋め込みベクトルを生成する（プロセス内、HTTP 不要）。

        Args:
            text: 埋め込む文字列

        Returns:
            384 次元の埋め込みベクトル（失敗時は None）
        """
        model = _load_st_model()
        if model is None:
            return None
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
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
        単一 FileChunk をベクトルインデックスに追加または更新する。
        summary が空の場合はスキップする。

        Args:
            chunk: 追加・更新する FileChunk

        Returns:
            成功した場合 True
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
            logger.warning("ChromaDB upsert エラー: %s — %s", chunk.path, exc)
            return False

    def needs_reembedding(self, chunk: FileChunk) -> bool:
        """
        指定チャンクが ChromaDB に存在しない、または mtime/size が変化している場合 True を返す。

        Args:
            chunk: チェックする FileChunk

        Returns:
            再埋め込みが必要な場合 True
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
            logger.warning("ChromaDB get エラー: %s — %s", chunk.path, exc)
            return True

    def migrate_from_chunks(self, chunks: List[FileChunk]) -> int:
        """
        既存の JSONL インデックスから ChromaDB へ一括移行する。
        summary が存在し、まだ埋め込まれていないチャンクのみ処理する。

        Args:
            chunks: 移行する FileChunk のリスト

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
        logger.info("ChromaDB 移行完了: %d 件を埋め込みました", embedded)
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
        ChromaDB が初期化されていない、または sentence-transformers が利用不可の場合は
        BM25 にフォールバックする。

        Args:
            all_chunks: 全 FileChunk のリスト（フォールバック用・パス解決用）
            query: 検索クエリ
            top_n: 返す件数

        Returns:
            上位 N 件の FileChunk リスト
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
            # sentence-transformers 未インストール時は BM25 に移譲
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

        if not found:
            return _bm25_fallback(all_chunks, query, top_n)

        return found


def _bm25_fallback(chunks: List[FileChunk], query: str, top_n: int) -> List[FileChunk]:
    """ChromaDB / sentence-transformers 未使用時の BM25 フォールバック。"""
    from localforge.infrastructure.bm25_adapter import get_top_chunks_bm25
    return get_top_chunks_bm25(chunks, query, top_n)
