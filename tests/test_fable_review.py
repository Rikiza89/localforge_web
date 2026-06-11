"""
Fable review 回帰テスト — レビューで修正したバグ・追加した機能の検証。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localforge.application.analysis_service import AnalysisService
from localforge.application.context_service import ContextService
from localforge.application.explanation_service import ExplanationService, REPORT_SECTIONS
from localforge.domain.models import ChunkStrategy, FileChunk, GenerationPlan, PlannedFile
from localforge.application.generation_service import reset_cancel
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient, pick_num_ctx


# =========================================================================
# SEARCH/REPLACE ブロック適用
# =========================================================================

class TestSearchReplaceBlocks:
    """_apply_search_replace_blocks の行アライメント・改行コード処理。"""

    def _apply(self, gen_svc, original, llm_output):
        return gen_svc._apply_search_replace_blocks(original, llm_output)

    @staticmethod
    def _block(search, replace):
        return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"

    def test_exact_match(self, generation_service):
        original = "def foo():\n    return 1\n"
        out = self._block("    return 1", "    return 2")
        result, applied, failed = self._apply(generation_service, original, out)
        assert applied == 1
        assert failed == []
        assert "return 2" in result

    def test_trailing_whitespace_line_fallback(self, generation_service):
        # ファイル側に行末スペースがあり、LLMは省略して出力するケース
        original = "line one   \nline two  \nline three\n"
        out = self._block("line one\nline two", "replaced\nlines")
        result, applied, failed = self._apply(generation_service, original, out)
        assert applied == 1
        assert "replaced\nlines\nline three" in result

    def test_midline_partial_match_does_not_corrupt(self, generation_service):
        # 行の途中のみ一致する SEARCH は行単位フォールバックで「失敗」になるべき。
        # （旧実装は文字オフセット一致で行全体を置換し "prefix " が消失していた）
        original = "prefix foo   \nbar   \nbaz\n"
        out = self._block("foo\nbar", "X\nY")
        result, applied, failed = self._apply(generation_service, original, out)
        assert applied == 0
        assert len(failed) == 1
        # LF正規化以外、内容は変わらない
        assert result.splitlines() == ["prefix foo   ", "bar   ", "baz"]

    def test_crlf_input_normalized_and_applied(self, generation_service):
        original = "alpha\r\nbeta\r\ngamma\r\n"
        out = self._block("beta", "BETA")
        result, applied, failed = self._apply(generation_service, original, out)
        assert applied == 1
        # 関数は LF 空間で返す（改行コードの復元は書き込み側の責務）
        assert "\r\n" not in result
        assert "BETA" in result


# =========================================================================
# 改行コード保存（CRLF / LF）
# =========================================================================

class TestNewlinePreservation:
    """編集後も元ファイルの改行コードが保存されることを確認する。"""

    def _regen(self, gen_svc, mock_llm, root, file_name, diff_output):
        reset_cancel()
        plan = GenerationPlan(
            project_name="t", description="t",
            files=[PlannedFile(path=file_name, description="d", dependencies=[])],
        )

        def mock_stream(model, prompt, system=None, **kwargs):
            yield diff_output

        mock_llm.stream_completion.side_effect = mock_stream
        return list(gen_svc.stream_regenerate_file(
            root=root, plan=plan, model="llama3.2",
            context_md="", file_path=file_name,
        ))

    def test_crlf_file_stays_crlf(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "crlf.py"
        target.write_bytes(b"# one\r\n# two\r\n")

        diff = "<<<<<<< SEARCH\n# one\n=======\n# ONE\n>>>>>>> REPLACE\n"
        events = self._regen(generation_service, mock_llm, tmp_path, "crlf.py", diff)
        assert any(e.get("done") for e in events)

        raw = target.read_bytes()
        assert b"# ONE\r\n" in raw
        assert b"\r\r\n" not in raw

    def test_lf_file_stays_lf(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "lf.py"
        target.write_bytes(b"# one\n# two\n")

        diff = "<<<<<<< SEARCH\n# one\n=======\n# ONE\n>>>>>>> REPLACE\n"
        events = self._regen(generation_service, mock_llm, tmp_path, "lf.py", diff)
        assert any(e.get("done") for e in events)

        raw = target.read_bytes()
        assert b"# ONE\n" in raw
        assert b"\r\n" not in raw


# =========================================================================
# BM25 アダプター
# =========================================================================

class TestBm25Adapter:
    """フィンガープリントキャッシュと退化スコアのフォールバック。"""

    @staticmethod
    def _chunk(path, content, mtime=1700000000.0):
        return FileChunk(
            path=path, content=content, strategy=ChunkStrategy.FULL,
            size=len(content), mtime=mtime, summary=content[:50],
        )

    def test_degenerate_scores_fall_back_to_keyword_count(self):
        # 2文書コーパスでは IDF が 0 に退化する — キーワードカウントで正しく順位付け
        from localforge.infrastructure.bm25_adapter import get_top_chunks_bm25
        c1 = self._chunk("sample.py", "def hello(): pass")
        c2 = self._chunk("auth.py", "def login(): pass")
        top = get_top_chunks_bm25([c1, c2], "login auth", top_n=1)
        assert top[0].path == "auth.py"

    def test_cache_invalidated_on_content_change(self):
        from localforge.infrastructure.bm25_adapter import get_top_chunks_bm25
        a1 = self._chunk("a.py", "alpha topic content", mtime=1.0)
        b1 = self._chunk("b.py", "unrelated stuff here", mtime=1.0)
        top = get_top_chunks_bm25([a1, b1], "alpha topic", top_n=1)
        assert top[0].path == "a.py"

        # 内容が入れ替わった新しいチャンクリスト（mtime も変化）→ キャッシュが再構築される
        a2 = self._chunk("a.py", "unrelated stuff here", mtime=2.0)
        b2 = self._chunk("b.py", "alpha topic content", mtime=2.0)
        top2 = get_top_chunks_bm25([a2, b2], "alpha topic", top_n=1)
        assert top2[0].path == "b.py"


# =========================================================================
# num_ctx バケット
# =========================================================================

class TestPickNumCtx:
    def test_small_prompt_uses_floor(self):
        assert pick_num_ctx(0) == 8192
        assert pick_num_ctx(2000) == 8192

    def test_large_prompt_scales_up(self):
        assert pick_num_ctx(10000) == 16384
        assert pick_num_ctx(30000) == 65536

    def test_cap(self):
        assert pick_num_ctx(10_000_000) == 131072


# =========================================================================
# セキュリティ: Host/Origin 検証・ピン留めパス検証
# =========================================================================

class TestHostOriginValidation:
    def test_valid_host_allowed(self, flask_client):
        resp = flask_client.get("/api/project/status", headers={"Host": "127.0.0.1:7331"})
        assert resp.status_code != 403

    def test_invalid_host_rejected(self, flask_client):
        resp = flask_client.get("/api/project/status", headers={"Host": "evil.example.com"})
        assert resp.status_code == 403

    def test_invalid_origin_rejected(self, flask_client):
        resp = flask_client.get(
            "/api/project/status",
            headers={"Host": "127.0.0.1:7331", "Origin": "http://evil.example.com"},
        )
        assert resp.status_code == 403

    def test_local_origin_allowed(self, flask_client):
        resp = flask_client.get(
            "/api/project/status",
            headers={"Host": "127.0.0.1:7331", "Origin": "http://127.0.0.1:7331"},
        )
        assert resp.status_code != 403


class TestPinnedPathValidation:
    def test_traversal_path_rejected(self, flask_client, tmp_path):
        project_svc = flask_client.application.config["project_service"]
        project_svc.open_project(tmp_path)

        resp = flask_client.post(
            "/api/project/pinned",
            json={"paths": ["../../etc/passwd"]},
        )
        assert resp.status_code == 403

    def test_valid_path_accepted(self, flask_client, tmp_path):
        project_svc = flask_client.application.config["project_service"]
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        project_svc.open_project(tmp_path)

        resp = flask_client.post(
            "/api/project/pinned",
            json={"paths": ["main.py"]},
        )
        assert resp.status_code == 200
        assert resp.get_json()["pinned"] == ["main.py"]


# =========================================================================
# 差分プレビュー（approve / reject）
# =========================================================================

class TestDiffPreview:
    def _preview(self, gen_svc, mock_llm, root, file_name, diff_output):
        reset_cancel()
        plan = GenerationPlan(
            project_name="t", description="t",
            files=[PlannedFile(path=file_name, description="d", dependencies=[])],
        )

        def mock_stream(model, prompt, system=None, **kwargs):
            yield diff_output

        mock_llm.stream_completion.side_effect = mock_stream
        return list(gen_svc.stream_regenerate_file(
            root=root, plan=plan, model="llama3.2",
            context_md="", file_path=file_name, preview=True,
        ))

    def test_preview_does_not_write(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("# old\n", encoding="utf-8")

        diff = "<<<<<<< SEARCH\n# old\n=======\n# new\n>>>>>>> REPLACE\n"
        events = self._preview(generation_service, mock_llm, tmp_path, "main.py", diff)

        previews = [e for e in events if "diff_preview" in e]
        assert len(previews) == 1
        assert "+# new" in previews[0]["diff_preview"]
        assert "-# old" in previews[0]["diff_preview"]
        # ディスクは未変更
        assert target.read_text(encoding="utf-8") == "# old\n"

    def test_apply_pending_edit_writes(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("# old\n", encoding="utf-8")

        diff = "<<<<<<< SEARCH\n# old\n=======\n# new\n>>>>>>> REPLACE\n"
        self._preview(generation_service, mock_llm, tmp_path, "main.py", diff)

        ok, msg = generation_service.apply_pending_edit(tmp_path, "main.py")
        assert ok, msg
        assert "# new" in target.read_text(encoding="utf-8")

        # 二重適用は失敗する
        ok2, _ = generation_service.apply_pending_edit(tmp_path, "main.py")
        assert not ok2

    def test_apply_refused_if_file_changed(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("# old\n", encoding="utf-8")

        diff = "<<<<<<< SEARCH\n# old\n=======\n# new\n>>>>>>> REPLACE\n"
        self._preview(generation_service, mock_llm, tmp_path, "main.py", diff)

        # プレビュー後にファイルが変更された
        target.write_text("# changed externally\n", encoding="utf-8")

        ok, msg = generation_service.apply_pending_edit(tmp_path, "main.py")
        assert not ok
        assert "変更" in msg
        assert target.read_text(encoding="utf-8") == "# changed externally\n"

    def test_discard_pending_edit(self, generation_service, mock_llm, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("# old\n", encoding="utf-8")

        diff = "<<<<<<< SEARCH\n# old\n=======\n# new\n>>>>>>> REPLACE\n"
        self._preview(generation_service, mock_llm, tmp_path, "main.py", diff)

        assert generation_service.discard_pending_edit(tmp_path, "main.py") is True
        assert target.read_text(encoding="utf-8") == "# old\n"
        # 破棄後に適用はできない
        ok, _ = generation_service.apply_pending_edit(tmp_path, "main.py")
        assert not ok


# =========================================================================
# レポートのセクション単位再生成（マージ保存）
# =========================================================================

class TestReportSectionMerge:
    @pytest.fixture
    def report_env(self, python_fixture_project):
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.cuda_available = False
        mock_llm.num_thread = None

        fs = FileSystemAdapter()
        index_adapter = IndexAdapter()
        context = ContextService()
        analysis_svc = AnalysisService(
            fs=fs, index_adapter=index_adapter, llm=mock_llm, context=context,
            semantic_cache_dir=python_fixture_project / "_semcache",
        )
        explanation_svc = ExplanationService(
            analysis=analysis_svc, llm=mock_llm, context=context,
        )
        # インデックス構築
        list(analysis_svc.build_index(root=python_fixture_project, model="llama3.2"))
        return python_fixture_project, explanation_svc, mock_llm

    def test_single_section_regen_preserves_others(self, report_env):
        root, explanation_svc, mock_llm = report_env

        def stream_old(model, prompt, system=None, **kwargs):
            yield "OLD-CONTENT"

        mock_llm.stream_completion.side_effect = stream_old
        list(explanation_svc.stream_report(root=root, model="llama3.2"))

        def stream_new(model, prompt, system=None, **kwargs):
            yield "NEW-CONTENT"

        mock_llm.stream_completion.side_effect = stream_new
        list(explanation_svc.stream_report(
            root=root, model="llama3.2", selected_section_indices=[2],
        ))

        sections = explanation_svc._load_existing_sections(root)
        # 再生成したセクションのみ NEW、他は OLD のまま
        assert sections[REPORT_SECTIONS[2]] == "NEW-CONTENT"
        assert sections[REPORT_SECTIONS[0]] == "OLD-CONTENT"
        assert sections[REPORT_SECTIONS[5]] == "OLD-CONTENT"
        assert len(sections) == len(REPORT_SECTIONS)
