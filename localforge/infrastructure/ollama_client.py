"""
OllamaのHTTP APIラッパー — SSEストリーミングに対応したOllamaクライアント実装。
LLMPortインターフェースを実装する唯一のクラス。
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Optional

import requests

from localforge.domain.exceptions import OllamaConnectionError, OllamaModelNotFoundError

logger = logging.getLogger(__name__)

# OllamaサーバーのデフォルトURL
_DEFAULT_BASE_URL = "http://localhost:11434"
# HTTPリクエストのタイムアウト秒数（ストリーミング時は別途設定）
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 120
# generate_sync 用タイムアウト（大型ローカルモデル向けに長めに設定）
_GENERATE_READ_TIMEOUT = 600


class OllamaClient:
    """
    Ollama APIに対するHTTPリクエストを管理するクライアントクラス。
    テキスト生成のストリーミングとモデル一覧取得をサポートする。
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        """
        OllamaClientを初期化する。

        Args:
            base_url: OllamaサーバーのベースURL
        """
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def is_available(self) -> bool:
        """
        Ollamaサーバーが起動していてアクセス可能かどうかを確認する。

        Returns:
            接続可能であればTrue
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=(_CONNECT_TIMEOUT, _CONNECT_TIMEOUT),
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        Ollamaで利用可能なモデルの一覧を返す。

        Returns:
            モデル名のリスト

        Raises:
            OllamaConnectionError: サーバーへの接続に失敗した場合
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Ollamaサーバーに接続できません: {exc}") from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"モデル一覧の取得に失敗しました: {exc}") from exc

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        read_timeout: int = _READ_TIMEOUT,
    ) -> Generator[str, None, None]:
        """
        Ollama generate APIを使用してテキストをストリーミング生成する。

        Args:
            model: 使用するOllamaモデル名
            prompt: ユーザープロンプト
            system: システムプロンプト（省略可能）

        Yields:
            テキストチャンク（文字列）

        Raises:
            OllamaConnectionError: サーバーへの接続に失敗した場合
            OllamaModelNotFoundError: 指定モデルが見つからない場合
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        logger.debug("Ollamaストリーミング開始: model=%s", model)

        try:
            with self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=(_CONNECT_TIMEOUT, read_timeout),
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"モデル '{model}' が見つかりません。"
                        f" `ollama pull {model}` で取得してください。"
                    )
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk_data = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        logger.warning("JSONデコードエラー: %s — 行: %r", exc, raw_line)
                        continue

                    token = chunk_data.get("response", "")
                    if token:
                        yield token

                    if chunk_data.get("done"):
                        logger.debug("Ollamaストリーミング完了")
                        break

        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Ollamaサーバーに接続できません: {exc}") from exc
        except (OllamaModelNotFoundError, OllamaConnectionError):
            raise
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Ollamaリクエストに失敗しました: {exc}") from exc

    def generate_sync(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """
        ストリーミングなしで完全なテキスト応答を生成する（テスト・内部用）。
        大型ローカルモデル向けに長めのタイムアウトを使用する。

        Args:
            model: 使用するOllamaモデル名
            prompt: ユーザープロンプト
            system: システムプロンプト（省略可能）

        Returns:
            生成されたテキスト全文

        Raises:
            OllamaConnectionError: サーバーへの接続に失敗した場合
            OllamaModelNotFoundError: 指定モデルが見つからない場合
        """
        return "".join(self.stream_completion(model, prompt, system, read_timeout=_GENERATE_READ_TIMEOUT))
