"""
VectorAdapter — ChromaDB + fastembed (ONNX) を使ったセマンティック検索アダプター。
.localforge/chroma/ に永続化する組み込み ChromaDB コレクションを管理する。

埋め込みバックエンド（優先順）:
  1. fastembed (BAAI/bge-small-en-v1.5, 384次元, ONNX インプロセス実行)
     → Ollama 不要、sentence-transformers より高速
  2. sentence-transformers (all-MiniLM-L6-v2, 384次元) — fastembed が利用不可の場合
  3. BM25 キーワード検索 — 両方とも利用不可の場合

モデルキャッシュ戦略 (fastembed):
  fastembed は HuggingFace hub 形式でモデルをキャッシュする。
  キャッシュディレクトリ: ./models/fastembed/
  モデルキャッシュ検出: models/fastembed/models--Qdrant--bge-small-en-v1.5-onnx-Q/ が存在するか確認
  ローカルキャッシュが存在する場合: HF_HUB_OFFLINE=1 を一時設定してネットワークアクセスを防ぐ
  ローカルキャッシュが存在しない場合: HuggingFace からダウンロードして models/fastembed/ に保存

プロキシ制限環境での対処:
  1. ネットワークがある環境で一度起動してモデルをダウンロードする
  2. ./models/fastembed/ フォルダごとプロキシ制限環境にコピーする
  3. コピー後は HF_HUB_OFFLINE=1 が自動設定されネットワーク不要で動作する

sentence-transformers フォールバックのキャッシュ戦略:
  1. LOCALFORGE_ST_MODEL_PATH が設定されている → そのパスから直接ロード
  2. ./models/all-MiniLM-L6-v2/ が存在する → そこからロード
  3. 上記いずれもない → HuggingFace からダウンロードして models/ に保存

SSL 証明書エラー（自己署名 CA / 企業プロキシ環境）への対処:
  LOCALFORGE_DISABLE_SSL=1 python main.py  （初回ダウンロード時のみ必要）
"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import List, Optional

from localforge.domain.models import FileChunk

logger = logging.getLogger(__name__)

_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "localforge_fastembed"
_CHROMA_DIR = "chroma"

# プロジェクトルートの models/ フォルダ（main.py と同じ階層）
_PROJECT_ROOT = Path(__file__).parent.parent.parent
# fastembed が HuggingFace hub 形式で保存するキャッシュディレクトリ
_LOCAL_FASTEMBED_CACHE = _PROJECT_ROOT / "models" / "fastembed"
# fastembed がモデルを保存する HF hub 形式のディレクトリ名
_FASTEMBED_HF_CACHE_NAME = "models--Qdrant--bge-small-en-v1.5-onnx-Q"
_LOCAL_ST_MODEL_DIR = _PROJECT_ROOT / "models" / _ST_MODEL_NAME

# モジュールレベルのモデルキャッシュ（プロセス内で一度だけロードする）
_embed_model_instance = None
_embed_model_load_attempted = False
_embed_backend: Optional[str] = None  # "fastembed" | "sentence-transformers" | None


# ---------------------------------------------------------------------------
# fastembed ロード
# ---------------------------------------------------------------------------

def _fastembed_cache_exists() -> bool:
    """
    fastembed の HuggingFace hub 形式キャッシュが存在するか確認する。
    既知のキャッシュディレクトリ名を優先し、見つからなければ models-- プレフィックスでも検索する。
    """
    if not _LOCAL_FASTEMBED_CACHE.is_dir():
        return False
    if (_LOCAL_FASTEMBED_CACHE / _FASTEMBED_HF_CACHE_NAME).is_dir():
        return True
    # フォールバック: models-- で始まるディレクトリが存在すれば HF hub キャッシュとみなす
    return any(
        p.is_dir() and p.name.startswith("models--")
        for p in _LOCAL_FASTEMBED_CACHE.iterdir()
    )


def _load_fastembed_model():
    """
    fastembed TextEmbedding モデルをロードする。
    ローカルキャッシュが存在する場合は HF_HUB_OFFLINE=1 を一時設定して
    HuggingFace へのネットワークアクセスを完全に防ぐ。
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        logger.debug("fastembed がインストールされていません。sentence-transformers を試みます。")
        return None

    env_path = os.environ.get("LOCALFORGE_FASTEMBED_MODEL_PATH", "").strip()
    if env_path and Path(env_path).is_dir():
        cache_dir = env_path
        logger.info("fastembed: 環境変数キャッシュディレクトリを使用: %s", cache_dir)
    else:
        cache_dir = str(_LOCAL_FASTEMBED_CACHE)

    _LOCAL_FASTEMBED_CACHE.mkdir(parents=True, exist_ok=True)

    # ローカルキャッシュが存在する場合はオフラインモードで起動してネットワークを使わない
    cache_found = _fastembed_cache_exists()
    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    if cache_found:
        logger.info("fastembed: ローカルキャッシュ検出 — オフラインモードで起動します: %s", cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("fastembed: ローカルキャッシュなし — HuggingFace からダウンロードします → %s", cache_dir)

    try:
        model = TextEmbedding(model_name=_FASTEMBED_MODEL_NAME, cache_dir=cache_dir)
        _ = list(model.embed(["test"]))
        logger.info("fastembed モデルのロード完了: %s", _FASTEMBED_MODEL_NAME)
        return model
    except Exception as exc:
        logger.warning("fastembed モデルのロードに失敗しました: %s", exc)
        return None
    finally:
        # HF_HUB_OFFLINE を元の状態に戻す
        if cache_found:
            if old_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline


# ---------------------------------------------------------------------------
# sentence-transformers フォールバック
# ---------------------------------------------------------------------------

def _is_ssl_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("ssl", "certificate", "cert", "client has been closed"))


def _load_st_model():
    """
    sentence-transformers モデルをロードしてキャッシュする（fastembed 失敗時のフォールバック）。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers がインストールされていません。BM25 検索のみ使用されます。"
        )
        return None

    env_path = os.environ.get("LOCALFORGE_ST_MODEL_PATH", "").strip()
    if env_path and Path(env_path).is_dir():
        load_from = env_path
        save_after_load = False
    elif _LOCAL_ST_MODEL_DIR.is_dir():
        load_from = str(_LOCAL_ST_MODEL_DIR)
        save_after_load = False
    else:
        load_from = _ST_MODEL_NAME
        save_after_load = True

    logger.info("sentence-transformers モデルをロード中 (フォールバック): %s", load_from)
    model = _try_load_st(SentenceTransformer, load_from)
    if model is None:
        _log_st_help()
        return None

    if save_after_load:
        _save_st_model_locally(model)

    return model


def _try_load_st(SentenceTransformer, load_from: str):
    try:
        model = SentenceTransformer(load_from)
        logger.info("sentence-transformers モデルのロード完了: %s", load_from)
        return model
    except Exception as exc:
        if not _is_ssl_error(exc):
            logger.warning("sentence-transformers モデルのロードに失敗しました: %s", exc)
            return None
        logger.warning("SSL エラーでモデルのロードに失敗しました: %s", exc)

    disable_ssl = os.environ.get("LOCALFORGE_DISABLE_SSL", "").lower() in ("1", "true", "yes")
    if not disable_ssl:
        logger.warning(
            "SSL 証明書エラーが発生しました。LOCALFORGE_DISABLE_SSL=1 を設定して"
            " 再起動するか、モデルをローカルに配置してください。"
        )
        return None

    logger.info("LOCALFORGE_DISABLE_SSL=1: SSL 検証を無効化してリトライします")
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    try:
        model = SentenceTransformer(load_from)
        logger.info("sentence-transformers モデルのロード完了 (SSL バイパス): %s", load_from)
        return model
    except Exception as exc2:
        logger.warning("SSL バイパス後もモデルのロードに失敗しました: %s", exc2)
        return None
    finally:
        ssl._create_default_https_context = _orig_ctx  # type: ignore[attr-defined]


def _save_st_model_locally(model) -> None:
    try:
        _LOCAL_ST_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(_LOCAL_ST_MODEL_DIR))
        os.environ["LOCALFORGE_ST_MODEL_PATH"] = str(_LOCAL_ST_MODEL_DIR)
        logger.info("sentence-transformers モデルを保存しました: %s", _LOCAL_ST_MODEL_DIR)
    except Exception as exc:
        logger.warning("モデルのローカル保存に失敗しました（動作には影響なし）: %s", exc)


def _log_st_help() -> None:
    logger.warning(
        "sentence-transformers も利用不可のため BM25 検索にフォールバックします。\n"
        "解決方法:\n"
        "  [A] fastembed モデルを事前配置:\n"
        "        URL: https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz\n"
        "        展開先: ./models/fastembed/fast-bge-small-en-v1.5/\n"
        "  [B] sentence-transformers を自動ダウンロード: python main.py\n"
        "  [C] BM25 のみを使用する（設定不要 — 現在このモードで動作中）"
    )


# ---------------------------------------------------------------------------
# 統合ロードエントリポイント
# ---------------------------------------------------------------------------

def _load_embed_model():
    """
    fastembed → sentence-transformers の順で埋め込みモデルをロードする。
    成功したバックエンド名を _embed_backend に記録する。
    """
    global _embed_model_instance, _embed_model_load_attempted, _embed_backend
    if _embed_model_load_attempted:
        return _embed_model_instance
    _embed_model_load_attempted = True

    model = _load_fastembed_model()
    if model is not None:
        _embed_model_instance = model
        _embed_backend = "fastembed"
        logger.info("埋め込みバックエンド: fastembed (BAAI/bge-small-en-v1.5)")
        return _embed_model_instance

    logger.info("fastembed 利用不可 — sentence-transformers にフォールバックします")
    model = _load_st_model()
    if model is not None:
        _embed_model_instance = model
        _embed_backend = "sentence-transformers"
        logger.info("埋め込みバックエンド: sentence-transformers (all-MiniLM-L6-v2)")
        return _embed_model_instance

    _embed_backend = None
    return None


# ---------------------------------------------------------------------------
# VectorAdapter
# ---------------------------------------------------------------------------

class VectorAdapter:
    """
    ChromaDB を使ったベクトルインデックスの永続化・検索を担うアダプター。
    fastembed (BAAI/bge-small-en-v1.5) でプロセス内埋め込みを生成する。
    fastembed が利用不可の場合は sentence-transformers (all-MiniLM-L6-v2) を使用する。
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
        model = _load_embed_model()
        if model is None:
            return None
        try:
            if _embed_backend == "fastembed":
                embedding = list(model.embed([text]))[0]
                return embedding.tolist()
            else:
                # sentence-transformers
                embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
                return embedding.tolist()
        except Exception as exc:
            logger.warning("埋め込み生成エラー: %s", exc)
            return None

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
        ChromaDB が初期化されていない、または埋め込みモデルが利用不可の場合は
        BM25 にフォールバックする。
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
