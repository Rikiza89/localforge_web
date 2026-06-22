"""
llama-server プロセスのライフサイクル管理。

LocalForge が `llama-server` を自前で起動・終了する場合に使用する（任意機能）。
既に稼働中のサーバーがある場合はそれに接続（アタッチ）し、新規起動はしない。

Vulkan ビルドの llama-server に対して `--n-gpu-layers` を指定すると、
CUDA を持たないマシンでも内蔵 GPU（Intel Arc 等）へレイヤーをオフロードできる。

環境変数:
  LLAMACPP_AUTO_START   "1" のとき本マネージャがサーバー起動を試みる（既定: 無効）
  LLAMACPP_BINARY       llama-server 実行ファイルのパス（既定: ./llamacpp/llama-server[.exe]）
  LLAMACPP_MODEL_PATH   ロードする GGUF モデルのパス（必須）
  LLAMACPP_SERVER_URL   接続先 URL（既定: http://127.0.0.1:8081）
  LLAMACPP_CTX          コンテキスト長 --ctx-size（既定: 16384）
  LLAMACPP_N_GPU_LAYERS GPU へオフロードするレイヤー数 --n-gpu-layers（既定: 0 = CPU のみ）
  LLAMACPP_THREADS      CPU スレッド数 --threads（既定: 物理コア数）
  LLAMACPP_EXTRA_ARGS   追加の起動引数（スペース区切り）
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def _truthy(val: Optional[str]) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


class LlamaServerManager:
    """llama-server サブプロセスの起動・ヘルスチェック・停止を管理する。"""

    def __init__(
        self,
        server_url: str,
        binary: Optional[str] = None,
        model_path: Optional[str] = None,
        ctx_size: int = 16384,
        n_gpu_layers: int = 0,
        threads: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._binary = binary
        self._model_path = model_path
        self._ctx_size = ctx_size
        self._n_gpu_layers = n_gpu_layers
        self._threads = threads
        self._extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "LlamaServerManager":
        """環境変数から設定を読み込んでインスタンスを生成する。"""
        from localforge.infrastructure.ollama_client import recommended_num_thread

        threads_env = os.environ.get("LLAMACPP_THREADS")
        threads = int(threads_env) if threads_env and threads_env.isdigit() else recommended_num_thread()

        ctx_env = os.environ.get("LLAMACPP_CTX", "16384")
        ngl_env = os.environ.get("LLAMACPP_N_GPU_LAYERS", "0")
        extra = shlex.split(os.environ.get("LLAMACPP_EXTRA_ARGS", ""))

        return cls(
            server_url=os.environ.get("LLAMACPP_SERVER_URL", "http://127.0.0.1:8081"),
            binary=os.environ.get("LLAMACPP_BINARY"),
            model_path=os.environ.get("LLAMACPP_MODEL_PATH"),
            ctx_size=int(ctx_env) if ctx_env.isdigit() else 16384,
            n_gpu_layers=int(ngl_env) if ngl_env.lstrip("-").isdigit() else 0,
            threads=threads,
            extra_args=extra,
        )

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """接続先 URL の /health が応答するか確認する。"""
        try:
            resp = requests.get(f"{self._server_url}/health", timeout=(5, 5))
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _resolve_binary(self) -> Optional[Path]:
        """llama-server 実行ファイルのパスを解決する。"""
        if self._binary:
            p = Path(self._binary)
            return p if p.exists() else None
        # 既定: プロジェクト直下の llamacpp/ を探す
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidate = Path.cwd() / "llamacpp" / exe
        return candidate if candidate.exists() else None

    def start(self, wait_timeout: float = 120.0) -> bool:
        """
        サーバーを起動する（必要な場合のみ）。

        - 既に /health が応答する場合はアタッチして True を返す（新規起動しない）。
        - バイナリ・モデルが見つからない場合は警告して False を返す。
        - 起動後 /health が ok になるまで最大 wait_timeout 秒待機する。

        Returns:
            サーバーが利用可能なら True
        """
        if self.is_running():
            logger.info("llama-server は既に稼働中です（アタッチ）: %s", self._server_url)
            return True

        binary = self._resolve_binary()
        if binary is None:
            logger.warning(
                "llama-server バイナリが見つかりません（LLAMACPP_BINARY 未設定 / llamacpp/ に不在）。"
                "外部で起動した llama-server に接続するか、バイナリを配置してください。"
            )
            return False
        if not self._model_path or not Path(self._model_path).exists():
            logger.warning(
                "LLAMACPP_MODEL_PATH のモデルが見つかりません: %s", self._model_path
            )
            return False

        port = urlsplit(self._server_url).port or 8081
        host = urlsplit(self._server_url).hostname or "127.0.0.1"
        args: List[str] = [
            str(binary),
            "-m", str(self._model_path),
            "--host", host,
            "--port", str(port),
            "--ctx-size", str(self._ctx_size),
            "--n-gpu-layers", str(self._n_gpu_layers),
        ]
        if self._threads:
            args += ["--threads", str(self._threads)]
        args += self._extra_args

        logger.info("llama-server を起動します: %s", " ".join(args))
        try:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.warning("llama-server の起動に失敗しました: %s", exc)
            return False

        # /health が ok になるまで待機
        start = time.time()
        while time.time() - start < wait_timeout:
            if self._proc.poll() is not None:
                logger.warning("llama-server プロセスが起動直後に終了しました")
                return False
            if self.is_running():
                logger.info("llama-server 起動完了: %s", self._server_url)
                return True
            time.sleep(0.5)
        logger.warning("llama-server の起動がタイムアウトしました（%.0fs）", wait_timeout)
        return False

    def stop(self) -> None:
        """本マネージャが起動したサーバープロセスのみ停止する。"""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            logger.info("llama-server を停止しました")
        except Exception as exc:
            logger.warning("llama-server の停止に失敗しました: %s", exc)
        finally:
            self._proc = None
