"""
HuggingFace Transformers クライアント — safetensors 形式のモデルを CPU で実行する。
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

_DEFAULT_MAX_NEW_TOKENS = 2048
_DEFAULT_TEMPERATURE = 0.7


def _ensure_transformers() -> None:
    try:
        import transformers  # noqa: F401
        import torch          # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "transformers または torch がインストールされていません。\n"
            f"pip install transformers torch accelerate を実行してください。詳細: {exc}"
        ) from exc


class HuggingFaceClient:
    """
    HuggingFace Transformers 経由で safetensors モデルをローカル実行する LLM クライアント。
    モデルは明示的に load_model() を呼んで初期化する必要がある。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._loaded_model_path: str = ""
        self._num_threads: int = 0  # 0 = auto

    # ------------------------------------------------------------------
    # LLMPort 実装
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            _ensure_transformers()
            return True
        except RuntimeError:
            return False

    def list_models(self) -> List[str]:
        return []  # hf_model_manager.scan_local_models() が担当

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        ロード済みモデルでテキストをストリーミング生成する。
        model パラメータは無視される（ロード済みモデルを使用する）。
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "HuggingFace モデルがロードされていません。先にモデルをロードしてください。"
            )

        import torch
        from transformers import TextIteratorStreamer

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            input_text = (
                f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
                if system else
                f"User: {prompt}\n\nAssistant:"
            )

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096,
        )

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        n_threads = self._num_threads or os.cpu_count() or 4
        torch.set_num_threads(n_threads)

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": _DEFAULT_MAX_NEW_TOKENS,
            "temperature": _DEFAULT_TEMPERATURE,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        thread = threading.Thread(
            target=self._model.generate, kwargs=gen_kwargs, daemon=True,
        )
        thread.start()

        try:
            for token_text in streamer:
                yield token_text
        except Exception as exc:
            logger.error("HuggingFace 推論エラー: %s", exc)
            raise RuntimeError(f"HuggingFace 推論に失敗しました: {exc}") from exc

    def unload_model(self, model: str = "") -> None:
        with self._lock:
            if self._model is None:
                return
            logger.info("HuggingFace モデルをアンロード: %s", self._loaded_model_path)
            del self._model
            self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            self._loaded_model_path = ""

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HuggingFace 固有の操作
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_dir: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
    ) -> None:
        """
        safetensors モデルディレクトリをメモリにロードする。

        Args:
            model_dir:  config.json を含むモデルディレクトリのパス
            n_ctx:      未使用（transformers は tokenizer の max_length で制御）
            n_threads:  CPU スレッド数（0 = 自動）
        """
        _ensure_transformers()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_dir)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"config.json が見つかりません。HuggingFace モデルディレクトリを指定してください: {model_dir}"
            )

        if self._model is not None:
            self.unload_model()

        if n_threads > 0:
            self._num_threads = n_threads

        logger.info("HuggingFace モデルをロード中: %s", model_dir)

        with self._lock:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(path), local_files_only=True,
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                self._model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                self._loaded_model_path = str(path)
                logger.info("HuggingFace モデルのロード完了: %s", model_dir)

            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._loaded_model_path = ""
                raise RuntimeError(f"モデルのロードに失敗しました: {exc}") from exc

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        self._num_threads = num_thread if num_thread is not None else 0

    def get_loaded_model_path(self) -> str:
        return self._loaded_model_path

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_vram_info(self) -> Optional[dict]:
        return None
