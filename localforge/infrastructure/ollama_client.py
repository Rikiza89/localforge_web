"""
OllamaのHTTP APIラッパー — SSEストリーミングに対応したOllamaクライアント実装。
LLMPortインターフェースを実装する唯一のクラス。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Generator, List, Optional

import requests

from localforge.domain.exceptions import OllamaConnectionError, OllamaModelNotFoundError

logger = logging.getLogger(__name__)

# OllamaサーバーのデフォルトURL（OLLAMA_HOST環境変数で上書き可能）
_DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# HTTPリクエストのタイムアウト秒数（ストリーミング時は別途設定）
# CPU推論はモデルロードが遅いため長めに設定する
_CONNECT_TIMEOUT = 30
# ストリーミング読み込みタイムアウト: 2時間（大規模プロジェクトの長時間生成に対応）
_READ_TIMEOUT = 7200
# generate_sync 用タイムアウト（バッチサマリー生成など同期呼び出し用）
_GENERATE_READ_TIMEOUT = 7200


def _detect_cuda() -> bool:
    """
    nvidia-smi を呼び出してCUDA対応GPUが存在するか確認する。
    nvidia-smi が見つからない・タイムアウト・エラーの場合はFalseを返す。
    外部ライブラリ不要。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


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

        self.cuda_available: bool = _detect_cuda()
        if self.cuda_available:
            logger.info("CUDA GPU検出: Ollamaリクエストにnum_gpu=-1を設定します")
        else:
            logger.info("CUDA GPU未検出: CPU推論モードで動作します")

        # CPUスレッド数（Noneでデフォルト自動設定）
        self.num_thread: Optional[int] = None

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        """
        Ollamaが使用するCPUスレッド数を設定する。
        Noneを渡すとOllamaのデフォルト（全コア自動）に戻す。

        Args:
            num_thread: 使用するCPUスレッド数（1以上の整数、またはNone）
        """
        self.num_thread = num_thread
        if num_thread is not None:
            logger.info("CPUスレッド数を設定: %d", num_thread)
        else:
            logger.info("CPUスレッド数をデフォルト（自動）にリセットしました")

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

    def is_model_loaded(self, model: str) -> Optional[bool]:
        """
        Ollama /api/ps を使って指定モデルが現在メモリに存在するか確認する。
        None: 確認できなかった（APIエラー等）
        True: ロード済み
        False: 未ロード（cold start が発生する）

        Args:
            model: 確認するモデル名

        Returns:
            True / False / None
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/api/ps",
                timeout=(_CONNECT_TIMEOUT, 5),
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            running = data.get("models", [])
            model_base = model.split(":")[0].lower()
            for m in running:
                name = m.get("name", m.get("model", "")).lower()
                if model_base in name or model.lower() in name:
                    return True
            return False
        except Exception:
            return None

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
        num_ctx: Optional[int] = None,  # Add num_ctx here
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

        # options: dict = {}
        # if self.cuda_available:
        #     options["num_gpu"] = -1          # 全レイヤーをGPUにオフロード
        # if self.num_thread is not None:
        #     options["num_thread"] = self.num_thread
        # if options:
        #     payload["options"] = options

        options: dict = {}
        if self.cuda_available:
            options["num_gpu"] = -1          # 全レイヤーをGPUにオフロード
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
            
        # --- NEW CODE START ---
        if num_ctx is not None:
            options["num_ctx"] = num_ctx     # トークンコンテキスト上限を適用
        # --- NEW CODE END ---
            
        if options:
            payload["options"] = options


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

                    # 思考モデル（Gemma, QwQ など）は "thinking" フィールドに推論トークンを持つ。
                    # \x01 プレフィックスで思考トークンをマークし、
                    # ルート層がメイン表示をスキップしてOllamaパネルにのみ転送する。
                    thinking = chunk_data.get("thinking", "")
                    if thinking:
                        yield f"\x01{thinking}"

                    if chunk_data.get("done"):
                        logger.debug("Ollamaストリーミング完了")
                        break

        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Ollamaサーバーに接続できません: {exc}") from exc
        except (OllamaModelNotFoundError, OllamaConnectionError):
            raise
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Ollamaリクエストに失敗しました: {exc}") from exc

    def unload_model(self, model: str) -> None:
        """
        Ollamaのkeep_alive=0を使ってモデルをVRAM/RAMから即時アンロードする。
        モデルが読み込まれていない・Ollamaが停止中など失敗しても警告のみでスキップする。

        Args:
            model: アンロードするOllamaモデル名
        """
        if not model:
            return
        try:
            # /api/generate だけでなく /api/chat にも keep_alive: 0 を送ることで、
            # どちらのAPIで読み込まれたモデルも確実にアンロードする。
            payload = {"model": model, "keep_alive": 0}
            self._session.post(
                f"{self._base_url}/api/generate",
                json={**payload, "prompt": ""},
                timeout=(_CONNECT_TIMEOUT, _CONNECT_TIMEOUT),
            )
            self._session.post(
                f"{self._base_url}/api/chat",
                json={**payload, "messages": []},
                timeout=(_CONNECT_TIMEOUT, _CONNECT_TIMEOUT),
            )
            logger.info("モデルをアンロードしました: %s", model)
        except Exception as exc:
            logger.warning("モデルアンロード失敗 (%s): %s", model, exc)

    def get_vram_info(self) -> Optional[dict]:
        """
        nvidia-smi を使用して現在のVRAM使用状況を取得する。
        CUDAが利用できない・エラーが発生した場合は None を返す。

        Returns:
            {"total": int, "used": int, "free": int} or None (単位: MiB)
        """
        if not self.cuda_available:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            # 最初の1行（最初のGPU）のみを取得
            line = result.stdout.strip().split("\n")[0]
            total, used, free = map(int, line.split(","))
            return {"total": total, "used": used, "free": free}
        except Exception:
            return None

    def get_ram_info(self) -> dict:
        """
        システムRAM使用状況を psutil で取得する。
        CPU専用デバイスではVRAMの代替として常に利用可能。

        Returns:
            {"total": int, "used": int, "free": int} (単位: MiB)
        """
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
        """
        GPU(VRAM)とシステムRAMの両方を返す複合エンドポイント用メソッド。
        GPUが存在しない場合は gpu フィールドを None にする。

        Returns:
            {"gpu": dict|None, "ram": dict, "cuda_available": bool}
        """
        return {
            "gpu": self.get_vram_info(),
            "ram": self.get_ram_info(),
            "cuda_available": self.cuda_available,
        }

    def generate_sync(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        num_ctx: Optional[int] = None,
        read_timeout: int = _GENERATE_READ_TIMEOUT,
    ) -> str:
        """
        ストリーミングなしで完全なテキスト応答を生成する（テスト・内部用）。
        大型ローカルモデル向けに長めのタイムアウトを使用する。
        read_timeout を指定することで CPU 推論での過長待機を防げる（例: 120s）。

        Args:
            model: 使用するOllamaモデル名
            prompt: ユーザープロンプト
            system: システムプロンプト（省略可能）
            num_ctx: コンテキスト長（省略可能）
            read_timeout: 読み込みタイムアウト秒数

        Returns:
            生成されたテキスト全文

        Raises:
            OllamaConnectionError: サーバーへの接続に失敗した場合
            OllamaModelNotFoundError: 指定モデルが見つからない場合
        """
        return "".join(self.stream_completion(
            model=model,
            prompt=prompt,
            system=system,
            read_timeout=read_timeout,
            num_ctx=num_ctx,
        ))
