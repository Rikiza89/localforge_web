"""
Explainパイプライン統合テスト — 15ファイルPythonフィクスチャでのテスト。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localforge.application.analysis_service import AnalysisService, _extract_python_landmarks
from localforge.application.context_service import ContextService
from localforge.application.explanation_service import ExplanationService, REPORT_SECTIONS
from localforge.domain.models import ChunkStrategy, FileChunk, Message, ProjectIndex
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient


class TestExplainPipelineE2E:
    """Explainパイプラインのエンドツーエンドテスト。"""

    @pytest.fixture
    def explain_env(self, python_fixture_project):
        """テスト環境を構築するフィクスチャ。"""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.cuda_available = False
        mock_llm.num_thread = None

        fs = FileSystemAdapter()
        index_adapter = IndexAdapter()
        context = ContextService()
        analysis_svc = AnalysisService(
            fs=fs,
            index_adapter=index_adapter,
            llm=mock_llm,
            context=context,
            semantic_cache_dir=python_fixture_project / "_semcache",
        )
        explanation_svc = ExplanationService(
            analysis=analysis_svc,
            llm=mock_llm,
            context=context,
        )

        return {
            "root": python_fixture_project,
            "analysis_svc": analysis_svc,
            "explanation_svc": explanation_svc,
            "mock_llm": mock_llm,
            "index_adapter": index_adapter,
        }

    def test_build_index_15_files(self, explain_env):
        """15ファイルのPythonフィクスチャでインデックス構築をテスト。"""
        root = explain_env["root"]
        analysis_svc = explain_env["analysis_svc"]

        events = list(analysis_svc.build_index(root=root, model="llama3.2"))

        progress_events = [e for e in events if "progress" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(done_events) == 1
        assert len(progress_events) >= 1

        # インデックスファイルが生成されたことを確認
        index_path = root / ".localforge" / "index.jsonl"
        assert index_path.exists()

        # 15件のチャンクが生成されたことを確認
        chunks = explain_env["index_adapter"].load_chunks(index_path)
        assert len(chunks) == 15

        # 各チャンクにサマリーが設定されていることを確認
        for chunk in chunks:
            assert chunk.summary is not None
            assert len(chunk.summary) > 0

    def test_incremental_reindex_skips_unchanged(self, explain_env):
        """増分インデックス: 変更なしファイルは再処理されないことをテスト。"""
        root = explain_env["root"]
        analysis_svc = explain_env["analysis_svc"]
        index_adapter = explain_env["index_adapter"]
        index_path = root / ".localforge" / "index.jsonl"

        # 1回目のインデックス構築
        list(analysis_svc.build_index(root=root, model="llama3.2"))
        first_indexed_at = {
            c.path: c.indexed_at for c in index_adapter.load_chunks(index_path)
        }

        # 2回目のインデックス構築（ファイル変更なし）
        list(analysis_svc.build_index(root=root, model="llama3.2"))
        second_indexed_at = {
            c.path: c.indexed_at for c in index_adapter.load_chunks(index_path)
        }

        # 変更なしファイルはキャッシュチャンクが再利用され、indexed_at が変わらないはず
        assert first_indexed_at == second_indexed_at

    def test_keyword_ranking(self, explain_env):
        """キーワードランキングでファイルが適切に選択されることをテスト。"""
        analysis_svc = explain_env["analysis_svc"]

        chunks = [
            FileChunk(
                path="auth/login.py",
                content="def login(user, password): ...",
                strategy=ChunkStrategy.FULL,
                size=30,
                mtime=1700000000.0,
                summary="ユーザーログイン機能",
            ),
            FileChunk(
                path="models/post.py",
                content="class Post: ...",
                strategy=ChunkStrategy.FULL,
                size=15,
                mtime=1700000000.0,
                summary="投稿モデル",
            ),
            FileChunk(
                path="auth/register.py",
                content="def register(user): ...",
                strategy=ChunkStrategy.FULL,
                size=25,
                mtime=1700000000.0,
                summary="ユーザー登録機能",
            ),
        ]

        top = analysis_svc._get_top_chunks_by_keywords(chunks, "login auth user", top_n=2)
        assert len(top) == 2
        paths = {c.path for c in top}
        assert "auth/login.py" in paths

    def test_stream_report_generates_all_sections(self, explain_env):
        """レポート生成が11セクションすべてをSSEで送信することをテスト。"""
        root = explain_env["root"]
        analysis_svc = explain_env["analysis_svc"]
        explanation_svc = explain_env["explanation_svc"]
        mock_llm = explain_env["mock_llm"]

        # インデックスを先に構築
        list(analysis_svc.build_index(root=root, model="llama3.2"))

        def mock_stream(model, prompt, system=None, **kwargs):
            yield "モックレポートテキスト。"

        mock_llm.stream_completion.side_effect = mock_stream

        events = list(explanation_svc.stream_report(root=root, model="llama3.2"))

        section_events = [e for e in events if "section" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(section_events) == len(REPORT_SECTIONS)
        assert len(done_events) == 1

        # すべてのセクション名が含まれることを確認
        section_names = {e["section"] for e in section_events}
        for expected_section in REPORT_SECTIONS:
            assert expected_section in section_names

    def test_stream_qa_answer(self, explain_env):
        """Q&A回答ストリーミングをテスト。"""
        root = explain_env["root"]
        analysis_svc = explain_env["analysis_svc"]
        explanation_svc = explain_env["explanation_svc"]
        mock_llm = explain_env["mock_llm"]

        # インデックスを先に構築
        list(analysis_svc.build_index(root=root, model="llama3.2"))

        def mock_qa_stream(model, prompt, system=None, **kwargs):
            yield "このプロジェクトはFlaskベースのWebアプリです。"

        mock_llm.stream_completion.side_effect = mock_qa_stream

        events = list(explanation_svc.stream_answer(
            root=root,
            model="llama3.2",
            question="このプロジェクトは何をするアプリですか？",
            history=[],
        ))

        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(token_events) >= 1
        assert len(done_events) == 1

    def test_qa_with_history(self, explain_env):
        """会話履歴ありのQ&Aをテスト。"""
        root = explain_env["root"]
        analysis_svc = explain_env["analysis_svc"]
        explanation_svc = explain_env["explanation_svc"]
        mock_llm = explain_env["mock_llm"]

        list(analysis_svc.build_index(root=root, model="llama3.2"))

        history = [
            Message(role="user", content="前の質問"),
            Message(role="assistant", content="前の回答"),
        ]

        def mock_qa_stream(model, prompt, system=None, **kwargs):
            # 会話履歴がプロンプトに含まれることを確認
            assert "前の質問" in prompt or "前の回答" in prompt
            yield "継続的な回答。"

        mock_llm.stream_completion.side_effect = mock_qa_stream

        events = list(explanation_svc.stream_answer(
            root=root,
            model="llama3.2",
            question="続けて教えてください",
            history=history,
        ))

        done_events = [e for e in events if e.get("done")]
        assert len(done_events) == 1


class TestPythonLandmarks:
    """Python ASTランドマーク抽出のテスト。"""

    def test_extracts_imports(self):
        source = "import os\nfrom pathlib import Path\n"
        landmarks = _extract_python_landmarks(source)
        assert any("import" in lm for lm in landmarks)

    def test_extracts_classes(self):
        source = "class MyClass:\n    pass\n"
        landmarks = _extract_python_landmarks(source)
        assert any("MyClass" in lm for lm in landmarks)

    def test_extracts_functions(self):
        source = "def my_function():\n    pass\n\nasync def async_func():\n    pass\n"
        landmarks = _extract_python_landmarks(source)
        assert any("function" in lm or "func" in lm for lm in landmarks)

    def test_handles_syntax_error_gracefully(self):
        # 構文エラーのあるコードでも例外が発生しないことを確認
        source = "def broken_function(:\n    pass"
        result = _extract_python_landmarks(source)
        assert isinstance(result, list)

    def test_no_duplicates(self):
        source = "import os\nimport sys\nfrom pathlib import Path\n"
        landmarks = _extract_python_landmarks(source)
        # 重複がないことを確認
        assert len(landmarks) == len(set(landmarks))
