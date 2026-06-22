"""
llama.cpp バックエンド + CPU 最適化（A2/A4）の回帰テスト。

llama-server への実 HTTP 接続は行わず、requests.Session をフェイクに差し替えて
ストリーミング解析・エンドポイント呼び出しを検証する。
"""

from __future__ import annotations

import json
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from localforge.infrastructure.llamacpp_client import LlamaCppClient
from localforge.infrastructure.llamacpp_server import LlamaServerManager, _truthy
from localforge.infrastructure.ollama_client import recommended_num_thread


# =========================================================================
# フェイク HTTP インフラ
# =========================================================================

class _FakeResp:
    """requests.Response の最小フェイク（context manager 対応）。"""

    def __init__(self, status_code=200, lines=None, json_data=None):
        self.status_code = status_code
        self._lines = lines or []
        self._json = json_data or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_lines(self):
        for ln in self._lines:
            yield ln if isinstance(ln, bytes) else ln.encode("utf-8")

    def json(self):
        return self._json


def _sse(obj: dict) -> str:
    """llama-server 風の SSE 行を作る。"""
    return f"data: {json.dumps(obj)}"


# =========================================================================
# LlamaCppClient.stream_completion
# =========================================================================

class TestLlamaCppStream:
    def test_parses_sse_content_and_stops(self):
        client = LlamaCppClient(base_url="http://x")
        lines = [
            _sse({"content": "Hello", "stop": False}),
            _sse({"content": " world", "stop": False}),
            _sse({"content": "", "stop": True}),
        ]
        captured = {}

        def fake_post(url, json=None, stream=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp(200, lines=lines)

        client._session = MagicMock()
        client._session.post.side_effect = fake_post

        out = list(client.stream_completion("any-model", "prompt text"))
        assert out == ["Hello", " world"]
        # /completion を叩く・cache_prompt が有効
        assert captured["url"].endswith("/completion")
        assert captured["payload"]["cache_prompt"] is True
        assert captured["payload"]["stream"] is True

    def test_num_predict_maps_to_n_predict(self):
        client = LlamaCppClient(base_url="http://x")
        captured = {}

        def fake_post(url, json=None, stream=None, timeout=None):
            captured["payload"] = json
            return _FakeResp(200, lines=[_sse({"content": "x", "stop": True})])

        client._session = MagicMock()
        client._session.post.side_effect = fake_post
        list(client.stream_completion("m", "p", num_predict=128))
        assert captured["payload"]["n_predict"] == 128

    def test_system_prompt_prepended(self):
        client = LlamaCppClient(base_url="http://x")
        captured = {}

        def fake_post(url, json=None, stream=None, timeout=None):
            captured["payload"] = json
            return _FakeResp(200, lines=[_sse({"content": "x", "stop": True})])

        client._session = MagicMock()
        client._session.post.side_effect = fake_post
        list(client.stream_completion("m", "USERPROMPT", system="SYSPROMPT"))
        assert captured["payload"]["prompt"].startswith("SYSPROMPT")
        assert "USERPROMPT" in captured["payload"]["prompt"]

    def test_reasoning_content_marked_with_x01(self):
        client = LlamaCppClient(base_url="http://x")
        lines = [
            _sse({"reasoning_content": "thinking...", "stop": False}),
            _sse({"content": "answer", "stop": True}),
        ]
        client._session = MagicMock()
        client._session.post.return_value = _FakeResp(200, lines=lines)
        out = list(client.stream_completion("m", "p"))
        assert "\x01thinking..." in out
        assert "answer" in out

    def test_connection_error_wrapped(self):
        import requests
        from localforge.domain.exceptions import OllamaConnectionError

        client = LlamaCppClient(base_url="http://x")
        client._session = MagicMock()
        client._session.post.side_effect = requests.ConnectionError("boom")
        with pytest.raises(OllamaConnectionError):
            list(client.stream_completion("m", "p"))


# =========================================================================
# LlamaCppClient — 補助メソッド
# =========================================================================

class TestLlamaCppHelpers:
    def test_is_available_true_on_200(self):
        client = LlamaCppClient(base_url="http://x")
        client._session = MagicMock()
        client._session.get.return_value = _FakeResp(200)
        assert client.is_available() is True

    def test_is_available_false_on_exception(self):
        import requests
        client = LlamaCppClient(base_url="http://x")
        client._session = MagicMock()
        client._session.get.side_effect = requests.ConnectionError()
        assert client.is_available() is False

    def test_list_models_from_v1_models(self):
        client = LlamaCppClient(base_url="http://x")
        client._session = MagicMock()
        client._session.get.return_value = _FakeResp(
            200, json_data={"data": [{"id": "qwen2.5-coder-7b"}]}
        )
        assert client.list_models() == ["qwen2.5-coder-7b"]

    def test_list_models_props_fallback(self):
        client = LlamaCppClient(base_url="http://x")
        client._session = MagicMock()
        # /v1/models は空、/props に model_path
        client._session.get.side_effect = [
            _FakeResp(200, json_data={"data": []}),
            _FakeResp(200, json_data={"model_path": "/models/foo-q4.gguf"}),
        ]
        assert client.list_models() == ["foo-q4.gguf"]

    def test_preload_returns_set_event(self):
        client = LlamaCppClient(base_url="http://x")
        ev = client.preload_model_async("m")
        assert ev.is_set() is True

    def test_unload_is_noop(self):
        client = LlamaCppClient(base_url="http://x")
        assert client.unload_model("m") is None

    def test_get_sysinfo_shape(self):
        client = LlamaCppClient(base_url="http://x")
        info = client.get_sysinfo()
        assert info["gpu"] is None
        assert info["cuda_available"] is False
        assert set(info["ram"].keys()) == {"total", "used", "free"}

    def test_cuda_available_is_false(self):
        assert LlamaCppClient(base_url="http://x").cuda_available is False

    def test_implements_llmport_surface(self):
        """ルート層が期待するメソッドが揃っていること（ドロップイン置換）。"""
        client = LlamaCppClient(base_url="http://x")
        for name in (
            "stream_completion", "list_models", "is_available", "unload_model",
            "set_num_thread", "get_sysinfo", "get_vram_info", "preload_model_async",
        ):
            assert callable(getattr(client, name)), name
        assert hasattr(client, "num_thread")


# =========================================================================
# LlamaServerManager
# =========================================================================

class TestLlamaServerManager:
    def test_from_env_defaults(self, monkeypatch):
        for k in ("LLAMACPP_SERVER_URL", "LLAMACPP_BINARY", "LLAMACPP_MODEL_PATH",
                  "LLAMACPP_CTX", "LLAMACPP_N_GPU_LAYERS", "LLAMACPP_THREADS",
                  "LLAMACPP_EXTRA_ARGS"):
            monkeypatch.delenv(k, raising=False)
        mgr = LlamaServerManager.from_env()
        assert mgr._server_url == "http://127.0.0.1:8081"
        assert mgr._ctx_size == 16384
        assert mgr._n_gpu_layers == 0

    def test_from_env_reads_gpu_layers(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_N_GPU_LAYERS", "33")
        monkeypatch.setenv("LLAMACPP_CTX", "32768")
        mgr = LlamaServerManager.from_env()
        assert mgr._n_gpu_layers == 33
        assert mgr._ctx_size == 32768

    def test_start_attaches_to_running_server(self, monkeypatch):
        mgr = LlamaServerManager(server_url="http://127.0.0.1:8081")
        monkeypatch.setattr(mgr, "is_running", lambda: True)
        # 既に稼働中ならバイナリ不要で True
        assert mgr.start() is True
        assert mgr._proc is None

    def test_start_returns_false_without_binary(self, monkeypatch):
        mgr = LlamaServerManager(server_url="http://127.0.0.1:18099")
        monkeypatch.setattr(mgr, "is_running", lambda: False)
        monkeypatch.setattr(mgr, "_resolve_binary", lambda: None)
        assert mgr.start() is False

    def test_stop_is_safe_when_not_started(self):
        LlamaServerManager(server_url="http://x").stop()  # 例外を出さない

    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("0", False), ("", False), (None, False), ("nope", False),
    ])
    def test_truthy(self, val, expected):
        assert _truthy(val) is expected


# =========================================================================
# バックエンド選択ファクトリ
# =========================================================================

class TestBackendFactory:
    def test_default_is_ollama(self, monkeypatch):
        import logging
        from localforge.interface.server import _build_llm_backend
        from localforge.infrastructure.ollama_client import OllamaClient
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        llm, mgr = _build_llm_backend(logging.getLogger("t"))
        assert isinstance(llm, OllamaClient)
        assert mgr is None

    def test_llamacpp_selected(self, monkeypatch):
        import logging
        from localforge.interface.server import _build_llm_backend
        monkeypatch.setenv("LLM_BACKEND", "llamacpp")
        monkeypatch.delenv("LLAMACPP_AUTO_START", raising=False)
        llm, mgr = _build_llm_backend(logging.getLogger("t"))
        assert isinstance(llm, LlamaCppClient)
        # AUTO_START 未設定なのでマネージャは起動しない
        assert mgr is None


# =========================================================================
# A2 / A4 — CPU 最適化
# =========================================================================

class TestCpuOptimizations:
    def test_recommended_num_thread_positive_or_none(self):
        n = recommended_num_thread()
        assert n is None or (isinstance(n, int) and n >= 1)

    def test_llm_options_no_num_predict_by_default(self, generation_service):
        opts = generation_service._llm_options(1000)
        assert "num_predict" not in opts
        assert "num_ctx" in opts

    def test_llm_options_includes_cap_when_set(self, generation_service):
        generation_service.set_max_output_tokens(2048)
        opts = generation_service._llm_options(1000)
        assert opts["num_predict"] == 2048

    def test_set_max_output_tokens_zero_disables(self, generation_service):
        generation_service.set_max_output_tokens(4096)
        generation_service.set_max_output_tokens(0)
        assert "num_predict" not in generation_service._llm_options(1000)
