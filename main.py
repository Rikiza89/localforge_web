"""
main.py — LocalForgeアプリケーションのエントリーポイント。
pywebviewウィンドウとFlaskサーバーを統合して起動する。
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import signal
import subprocess
import threading
import time

# sentence-transformers モデルをプロジェクト内の models/ フォルダから自動検出する。
# 環境変数が未設定の場合のみ適用する。
if not os.environ.get("LOCALFORGE_ST_MODEL_PATH"):
    _here = os.path.dirname(os.path.abspath(__file__))
    _model_dir = os.path.join(_here, "models", "all-MiniLM-L6-v2")
    if os.path.isdir(_model_dir):
        os.environ["LOCALFORGE_ST_MODEL_PATH"] = _model_dir

logger = logging.getLogger(__name__)

# Flask アプリへの参照（シグナルハンドラから HF クライアントにアクセスするため）
_flask_app = None


def _kill_ollama() -> None:
    """Ollamaプロセスを終了してVRAM/RAMを解放する。"""
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "ollama.exe"],
                capture_output=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ["pkill", "-x", "ollama"],
                capture_output=True,
                timeout=5,
            )
        logger.info("Ollamaプロセスを終了しました")
    except Exception as exc:
        logger.warning("Ollamaプロセス終了に失敗しました: %s", exc)


def _unload_hf_model() -> None:
    """HuggingFace モデルをアンロードしてRAMを解放する。"""
    global _flask_app
    try:
        if _flask_app is not None:
            hf_client = _flask_app.config.get("hf_client")
            if hf_client is not None and hf_client.is_model_loaded():
                hf_client.unload_model()
                logger.info("HuggingFace モデルをアンロードしました")
    except Exception as exc:
        logger.warning("HuggingFace モデルアンロードに失敗しました: %s", exc)


def _cleanup() -> None:
    """アプリ終了時にすべての LLM リソースを解放する。"""
    _unload_hf_model()
    _kill_ollama()


def _signal_handler(signum, frame) -> None:
    """SIGTERMまたはSIGINT受信時にすべての LLM リソースを解放してプロセスを終了する。"""
    logger.info("シグナル %d を受信 — LLM リソースを解放します", signum)
    _cleanup()
    os._exit(0)

# Flaskサーバーのホスト・ポート設定
# FLASK_HOST=0.0.0.0 にするとDockerコンテナ外からもアクセス可能（LAN公開）
_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
_PORT = int(os.environ.get("FLASK_PORT", "7331"))


def _check_network_exposure() -> None:
    """
    起動時にネットワーク公開リスクとなる環境変数を検査して警告する。
    FLASK_HOST や OLLAMA_HOST が外部アドレスに向いている場合、
    ファイル内容・LLMプロンプトが意図せずネットワークに出る可能性がある。
    """
    import re as _re
    _localhost_re = _re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?/?$")

    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host and not _localhost_re.match(ollama_host):
        logger.warning(
            "SECURITY WARNING: OLLAMA_HOST is set to an external address: %s — "
            "indexed file contents and LLM prompts will be sent to this host. "
            "Unset OLLAMA_HOST to use the local Ollama instance.",
            ollama_host,
        )

    if _HOST not in ("127.0.0.1", "localhost"):
        logger.warning(
            "SECURITY WARNING: FLASK_HOST is set to %s — "
            "the LocalForge API will be reachable from the network with no authentication. "
            "Unset FLASK_HOST to bind to localhost only.",
            _HOST,
        )


def start_flask(app) -> None:
    """
    Flaskサーバーを別スレッドで起動する関数。
    127.0.0.1のみにバインドしてセキュリティを確保する。

    Args:
        app: Flaskアプリケーションインスタンス
    """
    app.run(
        host=_HOST,
        port=_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """
    Flaskサーバーが起動するまで待機する。

    Args:
        host: サーバーホスト
        port: サーバーポート
        timeout: 最大待機時間（秒）

    Returns:
        サーバーが起動すればTrue、タイムアウトならFalse
    """
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


def main() -> None:
    """LocalForgeアプリケーションを起動する。"""
    from localforge.interface.server import create_app

    global _flask_app

    # 起動時にネットワーク公開リスクを検査する
    _check_network_exposure()

    # シグナルハンドラとatexitを登録（強制終了時にすべての LLM リソースを解放する）
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup)

    # Flaskアプリケーションを生成
    app = create_app()
    _flask_app = app

    # Flaskを別スレッドで起動
    flask_thread = threading.Thread(
        target=start_flask,
        args=(app,),
        daemon=True,
        name="FlaskServer",
    )
    flask_thread.start()
    logger.info("Flaskサーバースレッドを起動しました: http://%s:%d", _HOST, _PORT)

    # サーバーが起動するまで待機
    if not _wait_for_server(_HOST, _PORT):
        logger.error("Flaskサーバーの起動タイムアウト")
        raise RuntimeError("Flaskサーバーが起動しませんでした")

    # pywebviewウィンドウを開く（ネイティブOSウェブビュー）
    try:
        import webview

        window = webview.create_window(
            title="LocalForge",
            url=f"http://{_HOST}:{_PORT}",
            width=1280,
            height=800,
            min_size=(960, 600),
            resizable=True,
        )
        logger.info("pywebviewウィンドウを起動します")
        webview.start(debug=False)
        # ウィンドウが閉じられた後にすべての LLM リソースを解放する
        _cleanup()
    except Exception as exc:
        # ImportError: pywebviewが未インストール
        # WebViewException: GTK/QTがない環境（Docker・ヘッドレス）
        logger.warning(
            "pywebviewが利用できません (%s)。ブラウザで http://%s:%d を開いてください",
            type(exc).__name__, _HOST, _PORT,
        )
        print(f"\n✦ LocalForge が起動しました")
        print(f"  ブラウザで http://{_HOST}:{_PORT} を開いてください")
        print("  Ctrl+C で終了します\n")
        try:
            flask_thread.join()
        except KeyboardInterrupt:
            print("\n終了します")


if __name__ == "__main__":
    main()

