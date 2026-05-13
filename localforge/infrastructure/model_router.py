"""
ModelRouter — Ollama と HuggingFace クライアントへのディスパッチ層。
LLMPort を実装し、アクティブなプロバイダーに応じてリクエストを転送する。
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional

from localforge.infrastructure.ollama_client import OllamaClient
from localforge.infrastructure.huggingface_client import HuggingFaceClient

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Ollama / HuggingFace どちらかのクライアントにリクエストをルーティングする。
    server.py がすべてのサービスにこのクラスを注入する。
    """

    PROVIDER_OLLAMA = "ollama"
    PROVIDER_HF = "huggingface"

    def __init__(self, ollama: OllamaClient, hf: HuggingFaceClient) -> None:
        self._ollama = ollama
        self._hf = hf
        self._active_provider: str = self.PROVIDER_OLLAMA

    # ------------------------------------------------------------------
    # プロバイダー切替
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def switch_provider(self, provider: str) -> None:
        """
        アクティブなプロバイダーを切り替える。
        切替前に旧プロバイダーのモデルをアンロードしてメモリを解放する。

        Args:
            provider: "ollama" または "huggingface"
        """
        if provider not in (self.PROVIDER_OLLAMA, self.PROVIDER_HF):
            raise ValueError(f"不明なプロバイダー: {provider}")

        if provider == self._active_provider:
            return

        logger.info(
            "LLM プロバイダーを切替: %s → %s", self._active_provider, provider
        )

        # 旧プロバイダーのモデルをアンロード
        if self._active_provider == self.PROVIDER_OLLAMA:
            # アクティブな Ollama モデルを解放（モデル名は不明なためスキップ）
            pass
        elif self._active_provider == self.PROVIDER_HF:
            self._hf.unload_model()

        self._active_provider = provider
        logger.info("プロバイダー切替完了: %s", provider)

    # ------------------------------------------------------------------
    # LLMPort 実装 — アクティブプロバイダーに委譲
    # ------------------------------------------------------------------

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        return self._active_client().stream_completion(model, prompt, system)

    def list_models(self) -> List[str]:
        return self._active_client().list_models()

    def is_available(self) -> bool:
        return self._active_client().is_available()

    def unload_model(self, model: str = "") -> None:
        if self._active_provider == self.PROVIDER_OLLAMA:
            self._ollama.unload_model(model)
        else:
            self._hf.unload_model(model)

    # ------------------------------------------------------------------
    # Ollama 固有メソッド（プロジェクトルートが直接呼ぶもの）
    # ------------------------------------------------------------------

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        """両クライアントにスレッド数を適用する。"""
        self._ollama.set_num_thread(num_thread)
        self._hf.set_num_thread(num_thread)

    def get_vram_info(self) -> Optional[dict]:
        """GPU VRAM 情報を返す（Ollama のみ対応、HF は常に None）。"""
        if self._active_provider == self.PROVIDER_OLLAMA:
            return self._ollama.get_vram_info()
        return None

    def generate_sync(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """同期生成（Ollama のバッチ処理互換、HF はストリーム集積で実現）。"""
        if self._active_provider == self.PROVIDER_OLLAMA:
            return self._ollama.generate_sync(model, prompt, system, num_ctx)
        return "".join(self._hf.stream_completion(model, prompt, system))

    # ------------------------------------------------------------------
    # 直接アクセス（hf_routes.py から使用）
    # ------------------------------------------------------------------

    @property
    def ollama(self) -> OllamaClient:
        return self._ollama

    @property
    def hf(self) -> HuggingFaceClient:
        return self._hf

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _active_client(self):
        if self._active_provider == self.PROVIDER_OLLAMA:
            return self._ollama
        return self._hf

    # ------------------------------------------------------------------
    # num_thread 属性（project_routes が直接参照する）
    # ------------------------------------------------------------------

    @property
    def num_thread(self) -> Optional[int]:
        return self._ollama.num_thread

    @num_thread.setter
    def num_thread(self, value: Optional[int]) -> None:
        self._ollama.num_thread = value

    @property
    def cuda_available(self) -> bool:
        return self._ollama.cuda_available
