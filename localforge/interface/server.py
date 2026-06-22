"""
Flaskアプリケーション — ルート登録・依存性注入・ロギング設定を行うサーバーモジュール。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from flask import Flask

from localforge.application.analysis_service import AnalysisService
from localforge.application.context_service import ContextService
from localforge.application.explanation_service import ExplanationService
from localforge.application.generation_service import GenerationService
from localforge.application.project_service import ProjectService
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient, recommended_num_thread
from localforge.infrastructure.vector_adapter import VectorAdapter

# ログレベル設定
_LOG_FORMAT = (
    '{"time": "%(asctime)s", "level": "%(levelname)s",'
    ' "logger": "%(name)s", "message": "%(message)s"}'
)


def _configure_logging(log_dir: Path) -> None:
    """
    JSONフォーマットのローテーティングファイルログを設定する内部関数。

    Args:
        log_dir: ログファイルを配置するディレクトリ
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # コンソール出力も追加（開発時に有用）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def create_app(log_dir: Path = Path(".localforge")) -> Flask:
    """
    Flaskアプリケーションを生成して設定する。
    依存オブジェクトを構築し、ブループリントを登録する。

    Args:
        log_dir: ログファイルのディレクトリ（デフォルト: .localforge）

    Returns:
        設定済みのFlaskアプリケーション
    """
    _configure_logging(log_dir)
    logger = logging.getLogger(__name__)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    # ランダムなシークレットキーを生成する（セッション未使用だが念のため明示設定）
    app.config["SECRET_KEY"] = os.urandom(32)
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    # ---------------------------------------------------------------------------
    # インフラストラクチャ層のインスタンスを構築
    # ---------------------------------------------------------------------------
    fs = FileSystemAdapter()
    git = GitAdapter()
    index_adapter = IndexAdapter()
    # LLM バックエンドを LLM_BACKEND 環境変数で選択する（ollama 既定 / llamacpp）。
    # どちらも同じ LLMPort を実装するため、以降のサービス層は変更不要。
    llm, llamacpp_manager = _build_llm_backend(logger)
    vector = VectorAdapter()

    # CPU スレッド数の決定:
    #   1. LOCALFORGE_NUM_THREAD 環境変数（明示指定）が最優先
    #   2. 未指定かつ GPU 非搭載の場合は物理コア数を既定とする
    #      （論理コア全数だと SMT / E コアの過剰割当でスループットが落ちやすい）
    #   3. プロジェクトを開いた時点で config.json の num_thread が更にこれを上書きする
    _env_num_thread = os.environ.get("LOCALFORGE_NUM_THREAD")
    if _env_num_thread:
        try:
            llm.set_num_thread(int(_env_num_thread))
        except ValueError:
            logger.warning("LOCALFORGE_NUM_THREAD の値が不正です: %s", _env_num_thread)
    elif not getattr(llm, "cuda_available", False):
        _rec_threads = recommended_num_thread()
        if _rec_threads:
            llm.set_num_thread(_rec_threads)
            logger.info("CPU専用環境を検出: num_thread の既定値を物理コア数 %d に設定しました", _rec_threads)

    # ---------------------------------------------------------------------------
    # アプリケーション層のサービスを構築
    # ---------------------------------------------------------------------------
    context_svc = ContextService()
    project_svc = ProjectService(fs=fs, git=git, index=index_adapter)
    analysis_svc = AnalysisService(
        fs=fs, index_adapter=index_adapter, llm=llm, context=context_svc, vector=vector
    )
    explanation_svc = ExplanationService(
        analysis=analysis_svc, llm=llm, context=context_svc
    )
    generation_svc = GenerationService(
        fs=fs, git=git, index_adapter=index_adapter, llm=llm, context=context_svc
    )

    # LOCALFORGE_MAX_OUTPUT_TOKENS 環境変数で生成出力トークン上限を設定する（0 = 無制限）。
    # CPU推論で生成が暴走した場合のセーフティとして機能する。
    _env_max_out = os.environ.get("LOCALFORGE_MAX_OUTPUT_TOKENS")
    if _env_max_out:
        try:
            generation_svc.set_max_output_tokens(int(_env_max_out))
        except ValueError:
            logger.warning("LOCALFORGE_MAX_OUTPUT_TOKENS の値が不正です: %s", _env_max_out)

    # ---------------------------------------------------------------------------
    # サービスをアプリケーション設定に格納
    # ---------------------------------------------------------------------------
    app.config["project_service"] = project_svc
    app.config["generation_service"] = generation_svc
    app.config["analysis_service"] = analysis_svc
    app.config["explanation_service"] = explanation_svc
    app.config["context_service"] = context_svc
    app.config["llm"] = llm
    # "ollama_client" は後方互換のためのレガシーキー（実体は選択中の LLM クライアント）
    app.config["ollama_client"] = llm
    app.config["llamacpp_manager"] = llamacpp_manager
    app.config["git"] = git
    app.config["fs"] = fs
    app.config["vector"] = vector
    app.config["index_adapter"] = index_adapter

    # ---------------------------------------------------------------------------
    # ブループリントの登録
    # ---------------------------------------------------------------------------
    from localforge.interface.routes.project_routes import bp as project_bp
    from localforge.interface.routes.generation_routes import bp as generation_bp
    from localforge.interface.routes.explain_routes import bp as explain_bp
    from localforge.interface.routes.git_routes import bp as git_bp
    from localforge.interface.routes.workspace_routes import bp as workspace_bp

    app.register_blueprint(project_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(explain_bp)
    app.register_blueprint(git_bp)
    app.register_blueprint(workspace_bp)

    # ---------------------------------------------------------------------------
    # Host / Origin 検証（DNSリバインディング・CSRF対策）
    # ---------------------------------------------------------------------------
    # ループバック専用サーバーには認証がないため、ブラウザ経由の攻撃
    # （悪意あるWebページからの localhost へのリクエスト、DNSリバインディング）
    # を Host / Origin ヘッダー検証でブロックする。
    # FLASK_HOST を明示的に変更してLAN公開した場合は検証をスキップする
    # （main.py が起動時に警告を出す）。
    from urllib.parse import urlsplit

    from flask import abort, request

    _flask_host = os.environ.get("FLASK_HOST", "127.0.0.1").lower()
    _enforce_local = _flask_host in ("127.0.0.1", "localhost", "::1", "")
    _allowed_hosts = {"127.0.0.1", "localhost", "::1"}

    if not _enforce_local:
        logger.warning(
            "FLASK_HOST=%s のため Host/Origin 検証を無効化します。"
            "信頼できるネットワークでのみ使用してください。", _flask_host
        )

    @app.before_request
    def _validate_host_origin():
        if not _enforce_local:
            return None
        try:
            host = urlsplit("//" + (request.host or "")).hostname or ""
        except ValueError:
            abort(403)
        if host.lower() not in _allowed_hosts:
            logger.warning("不正な Host ヘッダーを拒否: %s", request.host)
            abort(403)
        origin = request.headers.get("Origin")
        if origin:
            try:
                origin_host = urlsplit(origin).hostname or ""
            except ValueError:
                origin_host = ""
            if origin_host.lower() not in _allowed_hosts:
                logger.warning("不正な Origin ヘッダーを拒否: %s", origin)
                abort(403)
        return None

    # ---------------------------------------------------------------------------
    # メインルート（SPAシェル）
    # ---------------------------------------------------------------------------
    from flask import render_template

    @app.route("/")
    def index():
        """メインSPAシェルを返す。"""
        return render_template("index.html")

    # ---------------------------------------------------------------------------
    # 起動時 LLM バックエンドヘルスチェック（Ollama / llama-server 共通）
    # ---------------------------------------------------------------------------
    if llm.is_available():
        try:
            models = llm.list_models()
            logger.info("LLMバックエンド接続確認: OK — 利用可能なモデル: %s", models)
        except Exception as exc:
            logger.warning("LLMバックエンド接続: 起動中だがモデル一覧取得失敗: %s", exc)
    else:
        logger.warning("LLMバックエンドに接続できません。Ollama または llama-server が起動しているか確認してください。")

    logger.info("LocalForge Flaskアプリケーション初期化完了")
    return app


def _build_llm_backend(logger: logging.Logger):
    """
    LLM_BACKEND 環境変数に基づいて LLM クライアント（LLMPort 実装）を構築する。

    Returns:
        (llm_client, llamacpp_manager or None)
    """
    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    if backend in ("llamacpp", "llama.cpp", "llama_cpp", "llama-cpp"):
        from localforge.infrastructure.llamacpp_client import LlamaCppClient
        from localforge.infrastructure.llamacpp_server import LlamaServerManager, _truthy

        client = LlamaCppClient()
        manager = None
        # LLAMACPP_AUTO_START=1 のときのみ llama-server プロセスを自動起動する。
        # 既定では外部で起動済みの llama-server へ接続する。
        if _truthy(os.environ.get("LLAMACPP_AUTO_START")):
            manager = LlamaServerManager.from_env()
            manager.start()
        logger.info("LLMバックエンド: llama.cpp (%s)", client._base_url)
        return client, manager

    logger.info("LLMバックエンド: Ollama")
    return OllamaClient(), None
