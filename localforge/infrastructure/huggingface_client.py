"""
HuggingFace GGUF モデルクライアント — llama-cpp-python を使用したローカル推論。
LLMPort インターフェースを実装する。GPU 不要、CPU のみで動作する。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

# llama-cpp-python の遅延インポート — インストールされていない場合はエラーを先送り
_llama_cpp = None
_llama_import_error: Optional[str] = None


def _ensure_llama_cpp() -> None:
    global _llama_cpp, _llama_import_error
    if _llama_cpp is not None:
        return
    try:
        import llama_cpp
        _llama_cpp = llama_cpp
    except ImportError as exc:
        _llama_import_error = (
            "llama-cpp-python がインストールされていません。"
            f" `pip install llama-cpp-python` を実行してください。詳細: {exc}"
        )
        raise RuntimeError(_llama_import_error) from exc


# デフォルトコンテキスト長（トークン数）— 32GB RAM 環境での推奨値
_DEFAULT_N_CTX = 8192
# デフォルト最大生成トークン数
_DEFAULT_MAX_TOKENS = 4096
# デフォルト CPU スレッド数（0 = llama.cpp が自動検出）
_DEFAULT_N_THREADS = 0


class HuggingFaceClient:
    """
    llama-cpp-python 経由で GGUF モデルをローカル実行する LLM クライアント。
    モデルは明示的に load_model() を呼んで初期化する必要がある。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llama = None          # llama_cpp.Llama インスタンス
        self._loaded_model_path: str = ""
        self._n_ctx: int = _DEFAULT_N_CTX
        self._n_threads: int = _DEFAULT_N_THREADS
        self._max_tokens: int = _DEFAULT_MAX_TOKENS

    # ------------------------------------------------------------------
    # LLMPort 実装
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """llama-cpp-python がインポート可能かどうかを返す。"""
        try:
            _ensure_llama_cpp()
            return True
        except RuntimeError:
            return False

    def list_models(self) -> List[str]:
        """~/.localforge/models/ 以下の GGUF ファイル名リストを返す。"""
        models_dir = Path.home() / ".localforge" / "models"
        if not models_dir.exists():
            return []
        result = []
        for gguf in sorted(models_dir.rglob("*.gguf")):
            result.append(str(gguf))
        return result

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        ロード済みモデルでテキストをストリーミング生成する。
        model パラメータは無視される（ロード済みモデルを使用する）。

        Yields:
            テキストチャンク（文字列）
        """
        with self._lock:
            if self._llama is None:
                raise RuntimeError(
                    "HuggingFace モデルがロードされていません。"
                    " 先に load_model() を呼び出してください。"
                )
            llama = self._llama

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug("HuggingFace ストリーミング開始: model=%s", self._loaded_model_path)

        try:
            for chunk in llama.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=self._max_tokens,
                temperature=0.7,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            logger.debug("HuggingFace ストリーミング完了")
        except Exception as exc:
            logger.error("HuggingFace 推論エラー: %s", exc)
            raise RuntimeError(f"HuggingFace 推論に失敗しました: {exc}") from exc

    def unload_model(self, model: str = "") -> None:
        """
        ロード済みモデルをメモリから解放する。
        model パラメータは無視される（常にロード済みモデルをアンロードする）。
        """
        with self._lock:
            if self._llama is None:
                return
            logger.info("HuggingFace モデルをアンロード: %s", self._loaded_model_path)
            try:
                del self._llama
            except Exception as exc:
                logger.warning("モデル削除中にエラー: %s", exc)
            self._llama = None
            self._loaded_model_path = ""

        gc.collect()
        # CUDA 環境では torch キャッシュも解放（インストール済みの場合のみ）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # HuggingFace 固有の操作
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_path: str,
        n_ctx: int = _DEFAULT_N_CTX,
        n_threads: int = _DEFAULT_N_THREADS,
    ) -> None:
        """
        GGUF ファイルをメモリにロードする。
        既にモデルがロードされている場合は先にアンロードする。

        Args:
            model_path: GGUF ファイルの絶対パス
            n_ctx:      コンテキスト長（トークン数）
            n_threads:  使用する CPU スレッド数（0 = 自動）
        """
        _ensure_llama_cpp()

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF ファイルが見つかりません: {model_path}")

        # 既存モデルをアンロード
        if self._llama is not None:
            self.unload_model()

        logger.info(
            "HuggingFace モデルをロード中: %s (n_ctx=%d, n_threads=%d)",
            model_path, n_ctx, n_threads or os.cpu_count() or 4,
        )

        with self._lock:
            self._llama = _llama_cpp.Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=n_threads if n_threads > 0 else (os.cpu_count() or 4),
                n_gpu_layers=0,     # CPU 専用 — GPU オフロードなし
                use_mlock=True,     # RAM ページをスワップアウトさせない
                verbose=False,
            )
            self._loaded_model_path = str(path)
            self._n_ctx = n_ctx
            self._n_threads = n_threads

        logger.info("HuggingFace モデルのロード完了: %s", model_path)

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        """CPU スレッド数を設定する（次回 load_model 時に適用）。"""
        self._n_threads = num_thread if num_thread is not None else _DEFAULT_N_THREADS

    def get_loaded_model_path(self) -> str:
        """現在ロード済みのモデルパスを返す。未ロードの場合は空文字列。"""
        return self._loaded_model_path

    def is_model_loaded(self) -> bool:
        """モデルがロード済みかどうかを返す。"""
        return self._llama is not None

    def get_vram_info(self) -> Optional[dict]:
        """HuggingFace クライアントは CPU 専用のため常に None を返す。"""
        return None
