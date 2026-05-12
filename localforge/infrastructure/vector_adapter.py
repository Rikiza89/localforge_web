"""
VectorAdapter — ChromaDB + sentence-transformers を使ったセマンティック検索アダプター。
.localforge/chroma/ に永続化する組み込み ChromaDB コレクションを管理する。

埋め込みモデルは sentence-transformers の all-MiniLM-L6-v2 (384 次元, ~25 MB) を
プロセス内で実行する。Ollama への HTTP 呼び出しは不要。
インクリメンタル更新をサポート: mtime+size が変化したファイルのみ再埋め込みする。

all-MiniLM-L6-v2 は初回起動時に HuggingFace Hub から自動ダウンロードされ、
~/.cache/huggingface/hub/ にキャッシュされる。

SSL 証明書エラー（自己署名 CA / 企業プロキシ環境）への対処:
  方法 1: ローカルにダウンロード済みのモデルを指定する
      LOCALFORGE_ST_MODEL_PATH=/path/to/all-MiniLM-L6-v2 python main.py
  方法 2: SSL 検証をスキップしてダウンロードする（初回のみ）
      LOCALFORGE_DISABLE_SSL=1 python main.py
  方法 3: ダウンロード後はキャッシュが使われるため以降は再設定不要
"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import List, Optional

from localforge.domain.models import FileChunk

logger = logging.getLogger(__name__)

_ST_MODEL_NAME = "all-MiniLM-L6-v2"
# nomic-embed-text (768 次元) との次元衝突を避けるため別コレクション名を使用する
_COLLECTION_NAME = "localforge_minilm"
_CHROMA_DIR = "chroma"

# モジュールレベルのモデルキャッシュ（プロセス内で一度だけロードする）
_st_model_instance = None
_st_model_load_attempted = False


def _is_ssl_error(exc: Exception) -> bool:
    """例外メッセージから SSL 証明書エラーか判定する。"""
    msg = str(exc).lower()
    return any(k in msg for k in ("ssl", "certificate", "cert", "client has been closed"))


def _load_st_model():
    """
    sentence-transformers モデルをロードしてキャッシュする。
    インストールされていない場合は None を返す（警告のみ、クラッシュしない）。

    優先順位:
      1. LOCALFORGE_ST_MODEL_PATH 環境変数が指すローカルパス
      2. 通常ダウンロード（HuggingFace Hub キャッシュ利用）
      3. SSL エラー時: LOCALFORGE_DISABLE_SSL=1 でリトライ
    """
    global _st_model_instance, _st_model_load_attempted
    if _st_model_load_attempted:
        return _st_model_instance
    _st_model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers がインストールされていません。"
            " ベクトル埋め込みは無効になり、BM25 検索のみ使用されます。"
            " 有効にするには: pip install sentence-transformers"
        )
        return None

    # LOCALFORGE_ST_MODEL_PATH が設定されていればローカルパスを使用
    model_path: str = os.environ.get("LOCALFORGE_ST_MODEL_PATH", "") or _ST_MODEL_NAME
    logger.info("sentence-transformers モデルをロード中: %s", model_path)

    # 1回目: 通常ロード
    try:
        _st_model_instance = SentenceTransformer(model_path)
        logger.info("sentence-transformers モデルのロード完了: %s", model_path)
        return _st_model_instance
    except Exception as exc:
        if not _is_ssl_error(exc):
            # SSL 以外のエラーはリトライしない
            logger.warning("sentence-transformers モデルのロードに失敗しました: %s", exc)
            _log_st_help()
            return None
        logger.warning("SSL エラーでモデルのダウンロードに失敗しました: %s", exc)

    # 2回目: SSL 検証を無効化してリトライ
    # LOCALFORGE_DISABLE_SSL=1 が設定されているか、自動リトライを有効にする
    disable_ssl = os.environ.get("LOCALFORGE_DISABLE_SSL", "").lower() in ("1", "true", "yes")
    if not disable_ssl:
        logger.warning(
            "SSL 証明書エラーが発生しました。自己署名 CA 環境では"
            " LOCALFORGE_DISABLE_SSL=1 を設定するか、モデルをローカルに配置してください。"
            " 詳細: vector_adapter.py のモジュール docstring を参照。"
        )
        _log_st_help()
        return None

    logger.info("LOCALFORGE_DISABLE_SSL=1: SSL 検証を無効化してリトライします")
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    try:
        _st_model_instance = SentenceTransformer(model_path)
        logger.info("sentence-transformers モデルのロード完了 (SSL バイパス): %s", model_path)
    except Exception as exc2:
        logger.warning("SSL バイパス後もモデルのロードに失敗しました: %s", exc2)
        _log_st_help()
        _st_model_instance = None
    finally:
        ssl._create_default_https_context = _orig_ctx  # type: ignore[attr-defined]

    return _st_model_instance


def _log_st_help() -> None:
    """SSL / ダウンロード失敗時の対処手順をログに出力する。"""
    logger.warning(
        "sentence-transformers が利用不可のため BM25 検索にフォールバックします。\n"
        "解決方法:\n"
        "  [A] SSL をスキップしてダウンロード（初回のみ）:\n"
        "        LOCALFORGE_DISABLE_SSL=1 python main.py\n"
        "  [B] ネットワークのある環境でモデルを事前保存してからローカルパスを指定:\n"
        "        python -c \"from sentence_transformers import SentenceTransformer;"
        " SentenceTransformer('all-MiniLM-L6-v2').save('/path/to/model')\"\n"
        "        LOCALFORGE_ST_MODEL_PATH=/path/to/model python main.py\n"
        "  [C] BM25 のみを使用する（設定不要 — 現在このモードで動作中）"
    )

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
            embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
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
