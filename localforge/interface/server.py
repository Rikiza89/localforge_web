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
from localforge.application.resume_service import ResumeService
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.huggingface_client import HuggingFaceClient
from localforge.infrastructure.model_router import ModelRouter
from localforge.infrastructure.ollama_client import OllamaClient
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
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    # ---------------------------------------------------------------------------
    # インフラストラクチャ層のインスタンスを構築
    # ---------------------------------------------------------------------------
    fs = FileSystemAdapter()
    git = GitAdapter()
    index_adapter = IndexAdapter()
    ollama_client = OllamaClient()
    hf_client = HuggingFaceClient()
    llm = ModelRouter(ollama=ollama_client, hf=hf_client)
    vector = VectorAdapter()

    # LOCALFORGE_NUM_THREAD 環境変数が設定されている場合は CPU スレッド数を適用する
    _env_num_thread = os.environ.get("LOCALFORGE_NUM_THREAD")
    if _env_num_thread:
        try:
            llm.set_num_thread(int(_env_num_thread))
        except ValueError:
            logger.warning("LOCALFORGE_NUM_THREAD の値が不正です: %s", _env_num_thread)

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
    resume_svc = ResumeService(
        fs=fs,
        git=git,
        generation=generation_svc,
        explanation=explanation_svc,
        context=context_svc,
    )

    # ---------------------------------------------------------------------------
    # サービスをアプリケーション設定に格納
    # ---------------------------------------------------------------------------
    app.config["project_service"] = project_svc
    app.config["generation_service"] = generation_svc
    app.config["analysis_service"] = analysis_svc
    app.config["explanation_service"] = explanation_svc
    app.config["resume_service"] = resume_svc
    app.config["context_service"] = context_svc
    app.config["llm"] = llm                 # ModelRouter インスタンス
    app.config["ollama_client"] = ollama_client
    app.config["hf_client"] = hf_client
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
    from localforge.interface.routes.hf_routes import bp as hf_bp
    from localforge.interface.routes.workspace_routes import bp as workspace_bp

    app.register_blueprint(project_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(explain_bp)
    app.register_blueprint(git_bp)
    app.register_blueprint(hf_bp)
    app.register_blueprint(workspace_bp)

    # ---------------------------------------------------------------------------
    # メインルート（SPAシェル）
    # ---------------------------------------------------------------------------
    from flask import render_template

    @app.route("/")
    def index():
        """メインSPAシェルを返す。"""
        return render_template("index.html")

    # ---------------------------------------------------------------------------
    # 起動時Ollamaヘルスチェック
    # ---------------------------------------------------------------------------
    if ollama_client.is_available():
        try:
            models = ollama_client.list_models()
            logger.info("Ollama接続確認: OK — 利用可能なモデル: %s", models)
        except Exception as exc:
            logger.warning("Ollama接続: サーバーは起動中だがモデル一覧取得失敗: %s", exc)
    else:
        logger.warning(
            "Ollama接続失敗: http://localhost:11434 に到達できません。"
            " HuggingFace プロバイダーは引き続き使用可能です。"
        )

    if hf_client.is_available():
        logger.info("HuggingFace (llama-cpp-python) 利用可能")
    else:
        logger.info("HuggingFace (llama-cpp-python) 未インストール — Ollama のみ使用可能")

    logger.info("LocalForge Flaskアプリケーション初期化完了")
    return app
