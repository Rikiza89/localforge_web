"""
テスト設定ファイル — pytestのフィクスチャとテスト共通設定を定義する。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock

import pytest

# テストからlocalforgeモジュールをインポートできるようにパスを設定
sys.path.insert(0, str(Path(__file__).parent.parent))

from localforge.application.context_service import ContextService
from localforge.application.generation_service import GenerationService
from localforge.application.analysis_service import AnalysisService
from localforge.application.explanation_service import ExplanationService
from localforge.application.project_service import ProjectService
from localforge.domain.models import (
    FileChunk,
    ChunkStrategy,
    GenerationPlan,
    PlannedFile,
    ProjectConfig,
    ProjectMode,
)
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """一時的なプロジェクトディレクトリを作成して返すフィクスチャ。"""
    return tmp_path


@pytest.fixture
def tmp_localforge_project(tmp_path: Path) -> Path:
    """
    .localforgeディレクトリを持つLocalForgeプロジェクトの
    一時ディレクトリを作成して返すフィクスチャ。
    """
    lf_dir = tmp_path / ".localforge"
    lf_dir.mkdir()
    config = ProjectConfig(
        project_name="test_project",
        mode=ProjectMode.GENERATE,
        model="llama3.2",
    )
    (lf_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def mock_llm() -> MagicMock:
    """
    OllamaClientのモックを返すフィクスチャ。
    stream_completionは文字列チャンクを順に返すジェネレーターを返す。
    """
    mock = MagicMock(spec=OllamaClient)
    mock.is_available.return_value = True
    mock.list_models.return_value = ["llama3.2", "codellama"]
    mock.cuda_available = False
    mock.num_thread = None

    def fake_stream(model, prompt, system=None, **kwargs):
        yield '{"project_name": "test", "description": "test", "files": []}'

    mock.stream_completion.side_effect = fake_stream
    return mock


@pytest.fixture
def fs_adapter() -> FileSystemAdapter:
    """FileSystemAdapterの実インスタンスを返すフィクスチャ。"""
    return FileSystemAdapter()


@pytest.fixture
def git_adapter() -> GitAdapter:
    """GitAdapterの実インスタンスを返すフィクスチャ。"""
    return GitAdapter()


@pytest.fixture
def index_adapter() -> IndexAdapter:
    """IndexAdapterの実インスタンスを返すフィクスチャ。"""
    return IndexAdapter()


@pytest.fixture
def context_service() -> ContextService:
    """ContextServiceの実インスタンスを返すフィクスチャ。"""
    return ContextService(token_limit=6000)


@pytest.fixture
def project_service(fs_adapter, git_adapter, index_adapter) -> ProjectService:
    """ProjectServiceの実インスタンスを返すフィクスチャ。"""
    return ProjectService(fs=fs_adapter, git=git_adapter, index=index_adapter)


@pytest.fixture
def generation_service(fs_adapter, git_adapter, index_adapter, mock_llm, context_service) -> GenerationService:
    """GenerationServiceのインスタンスを返すフィクスチャ（LLMはモック）。"""
    return GenerationService(
        fs=fs_adapter,
        git=git_adapter,
        index_adapter=index_adapter,
        llm=mock_llm,
        context=context_service,
    )


@pytest.fixture
def mock_vector() -> MagicMock:
    """
    VectorAdapterのモックを返すフィクスチャ。
    テスト時にChromaDBへの実際の接続を避けるために使用する。
    """
    from localforge.infrastructure.vector_adapter import VectorAdapter
    mock = MagicMock(spec=VectorAdapter)
    mock.needs_reembedding.return_value = True
    mock.upsert_chunk.return_value = True
    mock.collection_exists.return_value = True
    mock.get_top_chunks_semantic.return_value = []
    return mock


@pytest.fixture
def analysis_service(fs_adapter, index_adapter, mock_llm, context_service) -> AnalysisService:
    """AnalysisServiceのインスタンスを返すフィクスチャ（LLMはモック、Vectorなし）。"""
    return AnalysisService(
        fs=fs_adapter,
        index_adapter=index_adapter,
        llm=mock_llm,
        context=context_service,
        vector=None,
    )


@pytest.fixture
def analysis_service_with_vector(fs_adapter, index_adapter, mock_llm, context_service, mock_vector) -> AnalysisService:
    """VectorAdapterモックつきAnalysisServiceのインスタンスを返すフィクスチャ。"""
    return AnalysisService(
        fs=fs_adapter,
        index_adapter=index_adapter,
        llm=mock_llm,
        context=context_service,
        vector=mock_vector,
    )


@pytest.fixture
def sample_plan() -> GenerationPlan:
    """サンプルのGenerationPlanを返すフィクスチャ。"""
    return GenerationPlan(
        project_name="sample_app",
        description="サンプルプロジェクト",
        files=[
            PlannedFile(
                path="main.py",
                description="エントリーポイント",
                dependencies=[],
            ),
            PlannedFile(
                path="utils.py",
                description="ユーティリティ関数",
                dependencies=["main.py"],
            ),
        ],
    )


@pytest.fixture
def sample_chunk() -> FileChunk:
    """サンプルのFileChunkを返すフィクスチャ。"""
    return FileChunk(
        path="sample.py",
        content="def hello():\n    pass\n",
        strategy=ChunkStrategy.FULL,
        size=100,
        mtime=1700000000.0,
        summary="サンプルファイルのサマリー",
        language="python",
    )


@pytest.fixture
def flask_app():
    """テスト用のFlaskアプリケーションを返すフィクスチャ。"""
    import logging

    # ログディレクトリに一時ディレクトリを使用。
    # Windows では開いたままの app.log を削除できないため、teardown 時に
    # 一時ディレクトリ配下の FileHandler をすべて閉じてから削除する。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        from localforge.interface.server import create_app
        app = create_app(log_dir=Path(tmp_dir))
        app.config["TESTING"] = True
        yield app
        for handler in list(logging.getLogger().handlers):
            base = getattr(handler, "baseFilename", "")
            if base and str(base).startswith(tmp_dir):
                handler.close()
                logging.getLogger().removeHandler(handler)


@pytest.fixture
def flask_client(flask_app):
    """テスト用のFlaskクライアントを返すフィクスチャ。"""
    with flask_app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# 15ファイルのPython fixture（Explainパイプライン統合テスト用）
# ---------------------------------------------------------------------------

PYTHON_FIXTURE_FILES = {
    "main.py": 'from app import create_app\nif __name__ == "__main__":\n    app = create_app()\n    app.run()',
    "app.py": 'from flask import Flask\ndef create_app():\n    app = Flask(__name__)\n    return app',
    "models/user.py": 'class User:\n    def __init__(self, name: str):\n        self.name = name',
    "models/post.py": 'class Post:\n    def __init__(self, title: str, content: str):\n        self.title = title\n        self.content = content',
    "models/__init__.py": 'from .user import User\nfrom .post import Post',
    "routes/auth.py": 'def login():\n    pass\n\ndef logout():\n    pass',
    "routes/posts.py": 'def list_posts():\n    pass\n\ndef create_post():\n    pass',
    "routes/__init__.py": 'from .auth import login, logout\nfrom .posts import list_posts, create_post',
    "services/user_service.py": 'class UserService:\n    def get_user(self, user_id: int):\n        pass',
    "services/post_service.py": 'class PostService:\n    def get_posts(self):\n        return []\n    def create_post(self, title, content):\n        pass',
    "services/__init__.py": 'from .user_service import UserService\nfrom .post_service import PostService',
    "utils/helpers.py": 'def format_date(dt):\n    return dt.isoformat()\n\ndef slugify(text: str) -> str:\n    return text.lower().replace(" ", "-")',
    "utils/__init__.py": 'from .helpers import format_date, slugify',
    "config.py": 'class Config:\n    DEBUG = False\n    SECRET_KEY = "dev"\n\nclass TestConfig(Config):\n    TESTING = True',
    "tests/test_app.py": 'def test_app_creation():\n    from app import create_app\n    app = create_app()\n    assert app is not None',
}


@pytest.fixture
def python_fixture_project(tmp_path: Path) -> Path:
    """
    15ファイルのPythonコードベースを含む一時ディレクトリを作成して返すフィクスチャ。
    Explainパイプラインの統合テストに使用する。
    """
    for rel_path, content in PYTHON_FIXTURE_FILES.items():
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    return tmp_path
