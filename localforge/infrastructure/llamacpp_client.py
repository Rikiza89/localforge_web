"""
llama.cpp (llama-server) HTTP クライアント — LLMPort の代替実装。

Ollama の代わりに、ローカルで動作する `llama-server`（llama.cpp 同梱の HTTP サーバー）
に接続して推論を行う。CUDA を持たないマシンでも Vulkan バックエンドにより
内蔵 GPU（Intel Arc 等）へオフロードでき、純粋な CPU 推論より高速になりやすい。

設計方針:
- ネイティブ `/completion` エンドポイントを使用する（OpenAI 互換の
  `/v1/chat/completions` ではない）。LocalForge はプロンプトを自前で完全構築しており
  チャットテンプレートを使わないため、`/completion` に生プロンプトをそのまま渡すのが
  最も忠実で、かつ `cache_prompt`（KV プレフィックス再利用）が使える。
  これは 11 セクションのレポート生成（共通プレフィックス）や繰り返しの Q&A を加速する。
- llama-server は起動時に 1 モデルをロードする単一モデルサーバーであるため、
  `model` 引数はルーティングには使わない（ログ用途のみ）。
- `num_ctx` はサーバー起動時（--ctx-size）に固定されるため、リクエストごとの
  num_ctx は無視される（LLMPort の仕様どおり）。

LLMPort に加えて、ルート層が OllamaClient に期待する補助メソッド
（set_num_thread / num_thread / get_sysinfo / get_vram_info / preload_model_async /
is_model_loaded / cuda_available）も提供し、完全なドロップイン置換とする。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Generator, List, Optional

import requests

from localforge.domain.exceptions import OllamaConnectionError, OllamaModelNotFoundError

logger = logging.getLogger(__name__)

# llama-server のデフォルト URL（LLAMACPP_SERVER_URL で上書き可能）
_DEFAULT_SERVER_URL = os.environ.get("LLAMACPP_SERVER_URL", "http://127.0.0.1:8081")
_CONNECT_TIMEOUT = 30
# ストリーミング読み込みタイムアウト: 2時間（大規模生成に対応）
_READ_TIMEOUT = 7200


class LlamaCppClient:
    """
    llama-server に対する HTTP リクエストを管理するクライアント。
    OllamaClient と同じ LLMPort インターフェースを実装する。
    """

    def __init__(self, base_url: str = _DEFAULT_SERVER_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # llama.cpp は CUDA を直接使わない（Vulkan/CPU オフロード）。
        # アプリの _is_cpu ヒューリスティクスは cuda_available を見るため、
        # 常に False とし、安全側（小さめのコンテキスト）の挙動を選ばせる。
        self.cuda_available: bool = False
        # サーバー起動時の --threads に渡すための保持値（実行中サーバーには影響しない）。
        self.num_thread: Optional[int] = None

    # ------------------------------------------------------------------
    # LLMPort 実装
    # ------------------------------------------------------------------

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        """
        スレッド数を保持する。llama-server は起動時に --threads で固定されるため、
        既に稼働中のサーバーには即時反映されない（再起動時に使用される値）。
        """
        self.num_thread = num_thread
        logger.info("llama.cpp: num_thread を %s に設定（次回サーバー起動時に適用）", num_thread)

    def is_available(self) -> bool:
        """llama-server の /health が ok を返すか確認する。"""
        try:
            resp = self._session.get(
                f"{self._base_url}/health",
                timeout=(_CONNECT_TIMEOUT, 5),
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def is_model_loaded(self, model: str) -> Optional[bool]:
        """単一モデルサーバーなので /health が ok ならロード済みとみなす。"""
        return self.is_available() or None

    def list_models(self) -> List[str]:
        """
        ロード中のモデル名を返す。llama-server は単一モデルのため通常 1 件。
        /v1/models（OpenAI 互換）→ /props の順でフォールバックする。

        Raises:
            OllamaConnectionError: サーバーへ接続できない場合
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/v1/models",
                timeout=(_CONNECT_TIMEOUT, 10),
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                if models:
                    return models
        except requests.RequestException:
            pass

        # フォールバック: /props の model_path からファイル名を抽出
        try:
            resp = self._session.get(
                f"{self._base_url}/props",
                timeout=(_CONNECT_TIMEOUT, 10),
            )
            resp.raise_for_status()
            data = resp.json()
            model_path = (
                data.get("model_path")
                or data.get("default_generation_settings", {}).get("model")
                or ""
            )
            if model_path:
                return [os.path.basename(model_path)]
            return []
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"llama-server に接続できません: {exc}") from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"モデル一覧の取得に失敗しました: {exc}") from exc

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        read_timeout: int = _READ_TIMEOUT,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        keep_alive: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        llama-server の /completion でテキストをストリーミング生成する。

        Args:
            model: ログ用途のみ（単一モデルサーバーのためルーティングには未使用）
            prompt: ユーザープロンプト
            system: システムプロンプト（指定時は prompt の先頭に連結する）
            read_timeout: 読み込みタイムアウト秒数
            num_ctx: 無視される（コンテキスト長はサーバー起動時に固定）
            num_predict: 最大生成トークン数（-1 で無制限。/completion の n_predict にマップ）
            keep_alive: 無視される（単一モデルサーバー）

        Yields:
            テキストチャンク（文字列）。思考モデルの reasoning_content は
            \\x01 プレフィックス付きで yield する（Ollama 実装と同じ規約）。
        """
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload: dict = {
            "prompt": full_prompt,
            "stream": True,
            # KV プレフィックスキャッシュを有効化（共通プレフィックスの再計算を省く）
            "cache_prompt": True,
        }
        if num_predict is not None:
            payload["n_predict"] = num_predict

        logger.debug("llama.cpp ストリーミング開始: model=%s (参考)", model)

        try:
            with self._session.post(
                f"{self._base_url}/completion",
                json=payload,
                stream=True,
                timeout=(_CONNECT_TIMEOUT, read_timeout),
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(
                        "llama-server の /completion エンドポイントが見つかりません。"
                        " llama-server が起動しているか確認してください。"
                    )
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                    # SSE 形式: "data: {json}"。data: プレフィックスを剥がす。
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("JSONデコードエラー: %s — 行: %r", exc, line)
                        continue

                    token = chunk_data.get("content", "")
                    if token:
                        yield token

                    # 思考モデル（--reasoning-format 指定時）は reasoning_content を返す。
                    # Ollama 実装と同じく \x01 プレフィックスでマークしてルート層に委ねる。
                    reasoning = chunk_data.get("reasoning_content", "")
                    if reasoning:
                        yield f"\x01{reasoning}"

                    if chunk_data.get("stop"):
                        logger.debug("llama.cpp ストリーミング完了")
                        break

        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"llama-server に接続できません: {exc}") from exc
        except (OllamaModelNotFoundError, OllamaConnectionError):
            raise
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"llama-server リクエストに失敗しました: {exc}") from exc

    def preload_model_async(self, model: str) -> threading.Event:
        """
        llama-server は起動時に単一モデルをロード済みのため、プリロードは不要。
        即座に set 済みの Event を返し、呼び出し側が wait してもブロックしない。
        """
        ready = threading.Event()
        ready.set()
        return ready

    def unload_model(self, model: str) -> None:
        """
        単一モデルサーバーではアンロードという概念がないため no-op。
        （モデルの解放はサーバープロセス停止で行う — LlamaServerManager 参照）
        """
        return None

    # ------------------------------------------------------------------
    # システム情報（ルート層 /api/project/sysinfo 用）
    # ------------------------------------------------------------------

    def get_vram_info(self) -> Optional[dict]:
        """llama.cpp 経由では NVIDIA VRAM 情報を取得しない（None）。"""
        return None

    def get_ram_info(self) -> dict:
        """システム RAM 使用状況を psutil で取得する。"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            to_mib = 1024 * 1024
            return {
                "total": mem.total // to_mib,
                "used": mem.used // to_mib,
                "free": mem.available // to_mib,
            }
        except Exception:
            return {"total": 0, "used": 0, "free": 0}

    def get_sysinfo(self) -> dict:
        """GPU(VRAM) と RAM をまとめて返す。llama.cpp では gpu=None。"""
        return {
            "gpu": None,
            "ram": self.get_ram_info(),
            "cuda_available": self.cuda_available,
        }
