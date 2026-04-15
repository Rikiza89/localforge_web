"""
main.py — LocalForgeアプリケーションのエントリーポイント。
pywebviewウィンドウとFlaskサーバーを統合して起動する。
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Flaskサーバーのホスト・ポート設定
_HOST = "127.0.0.1"  # セキュリティ上の理由でローカルホストのみにバインド
_PORT = 7331


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


if __name__ == "__main__":
    from localforge.interface.server import create_app

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
