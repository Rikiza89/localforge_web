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

logger = logging.getLogger(__name__)


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


def _signal_handler(signum, frame) -> None:
    """SIGTERMまたはSIGINT受信時にOllamaを終了してプロセスを終了する。"""
    logger.info("シグナル %d を受信 — Ollamaを終了します", signum)
    _kill_ollama()
    os._exit(0)

# Flaskサーバーのホスト・ポート設定
# FLASK_HOST=0.0.0.0 にするとDockerコンテナ外からもアクセス可能（LAN公開）
_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
_PORT = int(os.environ.get("FLASK_PORT", "7331"))


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

    # シグナルハンドラとatexitを登録（強制終了時にOllamaを確実に終了させる）
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_kill_ollama)

    # Flaskアプリケーションを生成
    app = create_app()

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
        # ウィンドウが閉じられた後にOllamaを終了する
        _kill_ollama()
    except ImportError:
        # pywebviewが利用できない環境（テスト・CI）ではブラウザで開く
        logger.warning(
            "pywebviewが見つかりません。ブラウザで http://%s:%d を開いてください", _HOST, _PORT
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

