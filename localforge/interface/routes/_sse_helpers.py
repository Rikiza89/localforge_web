"""
Shared SSE utilities used by all route blueprints.
"""
from __future__ import annotations

import json
import queue
import threading

from flask import Response, current_app, jsonify, stream_with_context

from localforge.application.analysis_service import AnalysisService
from localforge.application.generation_service import GenerationService
from localforge.application.project_service import ProjectService
from localforge.infrastructure.git_adapter import GitAdapter

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream",
}

_HEARTBEAT_INTERVAL = 15  # seconds

_HB = {"heartbeat": True}


def _sse_response(generator):
    """
    Wrap a generator in a heartbeated SSE Response.
    A background thread injects heartbeat events every 15 s so the stream
    stays alive even when the LLM generator is blocked.
    """
    def wrapped():
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def _produce():
            try:
                for payload in generator:
                    if stop.is_set():
                        break
                    q.put(payload)
            except Exception as exc:
                q.put({"error": str(exc)})
            finally:
                # クライアント切断（stop）時もジェネレーターの finally を確実に実行し、
                # Ollama への HTTP ストリームを閉じて生成を停止させる
                try:
                    generator.close()
                except Exception:
                    pass
                q.put(None)  # sentinel

        def _heartbeat():
            while not stop.wait(_HEARTBEAT_INTERVAL):
                q.put(_HB)

        threading.Thread(target=_produce, daemon=True).start()
        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            while True:
                payload = q.get()
                if payload is None:
                    break
                if "token" in payload:
                    tok = payload["token"]
                    if tok.startswith("\x01"):
                        thinking_text = tok[1:]
                        yield f"data: {json.dumps({'raw_token': '<think>' + thinking_text + '</think>'}, ensure_ascii=False)}\n\n"
                        continue
                    else:
                        yield f"data: {json.dumps({'raw_token': tok}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            stop.set()

    return Response(
        stream_with_context(wrapped()),
        mimetype="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _error_response(exc: Exception, status: int = 500):
    return jsonify({"error": type(exc).__name__, "message": str(exc)}), status


def _get_project_svc() -> ProjectService:
    return current_app.config["project_service"]


def _get_generation_svc() -> GenerationService:
    return current_app.config["generation_service"]


def _get_analysis_svc() -> AnalysisService:
    return current_app.config["analysis_service"]


def _get_git() -> GitAdapter:
    return current_app.config["git"]
