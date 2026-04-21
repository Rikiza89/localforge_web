# LocalForge — Claude Code Skills Reference

This file describes repeatable implementation patterns for common extension tasks.
Each skill maps to a concrete sequence of file changes.

---

## Skill: Add a New Generation Mode

**When**: You want a new top-level workflow beyond Generate / Resume / Explain.

**Files to change**:
1. `domain/models.py` — add value to `ProjectMode` enum
2. `domain/ports.py` — add any new port interfaces needed
3. `application/<mode>_service.py` — create service with streaming generator methods
4. `interface/routes/<mode>_routes.py` — create Flask blueprint, use `_sse_response(generator)`
5. `interface/server.py` — instantiate service, register blueprint
6. `interface/templates/base.html` — add `<button class="tab-btn" data-tab="<mode>">` in `<nav class="tab-bar">`
7. `interface/templates/partials/<mode>_panel.html` — create partial
8. `interface/templates/base.html` — add `<div class="tab-content" id="tab-content-<mode>">{% include ... %}</div>` in `<main>`
9. `interface/static/js/app.js` — add tab handler and mode functions in `DOMContentLoaded`

**Pattern for streaming route**:
```python
@bp.route("/start", methods=["GET"])
def stream_start():
    project = _get_project_svc().current_project
    if not project:
        def err(): yield {"error": "No project"}
        return _sse_response(err())
    gen = _get_mode_svc().stream_operation(root=project.root, model=project.config.model)
    return _sse_response(gen)
```

---

## Skill: Add a New Report Section

**When**: You want to add a 12th (or more) section to the Explain report.

**Files to change**:
1. `application/explanation_service.py` — add the section name to `_REPORT_SECTIONS` list
2. `application/context_service.py` — optionally add a specialised `build_report_section_prompt` variant if the section needs a unique prompt structure

**Key list** (in `explanation_service.py`):
```python
_REPORT_SECTIONS = [
    "アーキテクチャ概要",
    "主要コンポーネント",
    # ... add new section name here
]
```

The report loop in `stream_report()` iterates this list automatically — no other changes needed.

---

## Skill: Add a New Infrastructure Adapter

**When**: You need a new external integration (database, cloud API, cache, etc.).

**Steps**:
1. Define a `Protocol` in `domain/ports.py`:
```python
class MyPort(Protocol):
    def do_thing(self, arg: str) -> str: ...
```
2. Implement in `infrastructure/my_adapter.py`:
```python
class MyAdapter:
    def do_thing(self, arg: str) -> str:
        # real I/O here
        ...
```
3. In `interface/server.py` `create_app()`:
```python
from localforge.infrastructure.my_adapter import MyAdapter
my_adapter = MyAdapter()
# inject into services that need it
app.config["my_adapter"] = my_adapter
```
4. Add parameter to the service `__init__` that needs it, store as `self._my = my_adapter`

---

## Skill: Extend the RAG Vector Index

**When**: You want to change embedding model, vector DB, or search strategy.

**Key files**:
- `infrastructure/vector_adapter.py` — all ChromaDB and Ollama embedding logic
- `application/analysis_service.py` — calls `self._vector.upsert_chunk(chunk)` after each file is summarised
- `application/analysis_service.py:get_top_chunks_semantic()` — single entry point for all semantic search

**To change embedding model**:
- Update `_EMBED_MODEL` constant in `vector_adapter.py`
- Delete `.localforge/chroma/` in existing projects to force re-embedding on next index run

**To add metadata filters** (e.g., search only Python files):
- Update `VectorAdapter.get_top_chunks_semantic()` to pass `where={"language": "python"}` to `self._collection.query()`

**To add hybrid search** (semantic + keyword reranking):
- In `get_top_chunks_semantic()`, after getting ChromaDB results, apply keyword scoring on top-K and re-sort

---

## Skill: Debug Streaming / Timeout Issues

**Symptoms**: "stream idle timeout", partial responses, frozen progress bar.

**Checklist**:
1. **Heartbeat**: `_sse_response()` in both route files emits `{"heartbeat": true}` every 15s. Verify it's present in both `generation_routes.py` and `explain_routes.py`.
2. **Client idle timer**: `startStream()` in `stream.js` reconnects after 30s without any event. Check `IDLE_TIMEOUT = 30000`.
3. **Large batches**: `analysis_service.py` saves JSONL every `_INCREMENTAL_SAVE_INTERVAL = 50` files. For very slow LLMs, this may still trigger timeouts — reduce to 20.
4. **Ollama read timeout**: `_READ_TIMEOUT = 120` in `ollama_client.py`. For very large models, increase to 300+.
5. **Flask buffering**: `X-Accel-Buffering: no` header is set in `_SSE_HEADERS`. If behind nginx, ensure `proxy_buffering off` is set.

---

## Skill: Add a New SSE Event Type

**When**: You need the frontend to react to a new kind of backend event.

**Backend** (in service generator method):
```python
yield {"my_event": {"key": "value"}}
```

**Frontend** (`stream.js` in `_dispatch()` and `startPostStream()` inner loop):
```javascript
if (data.my_event !== undefined && handlers.onMyEvent) {
    handlers.onMyEvent(data.my_event);
}
```

**Call site** (`app.js`):
```javascript
startStream("/api/endpoint", null, {
    onMyEvent: (payload) => { /* handle it */ },
    onDone: () => { ... },
});
```

---

## Skill: Tune Token Budget

**When**: LLM responses are getting cut off or prompts are too large.

**Per-project**: Edit `.localforge/config.json`:
```json
{"token_limit": 12000}
```

**Global default**: `application/context_service.py`:
```python
_DEFAULT_TOKEN_LIMIT = 6000  # increase here
```

**Batch summarisation budget**: `application/analysis_service.py`:
```python
_TOKEN_BUDGET_PER_BATCH = 3500  # tokens per LLM batch during indexing
```

Token estimation formula: `words × 1.3 ≈ tokens` (defined in `context_service._estimate_tokens()`).
