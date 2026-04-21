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
- `application/analysis_service.py` — collects chunks in `embed_queue`, then embeds in parallel with `ThreadPoolExecutor(max_workers=4)` after JSONL indexing completes
- `application/analysis_service.py:get_top_chunks_semantic()` — single entry point for all semantic search

**To change embedding model**:
- Update `_EMBED_MODEL` constant in `vector_adapter.py`
- Delete `.localforge/chroma/` in existing projects to force re-embedding on next index run

**To change embedding parallelism**:
- Update `max_workers=4` in the `ThreadPoolExecutor` call inside `build_index()` in `analysis_service.py`
- Higher values speed up embedding but increase Ollama memory pressure

**To add metadata filters** (e.g., search only Python files):
- Update `VectorAdapter.get_top_chunks_semantic()` to pass `where={"language": "python"}` to `self._collection.query()`

**To add hybrid search** (semantic + keyword reranking):
- In `get_top_chunks_semantic()`, after getting ChromaDB results, apply keyword scoring on top-K and re-sort

**Embedding flow** (as of current implementation):
1. Tier-0/1/2 chunks are collected into `embed_queue` during JSONL indexing (non-blocking)
2. Cached JSONL chunks missing from ChromaDB are detected via `needs_reembedding()` and added to `embed_queue`
3. All queued chunks are embedded in parallel after JSONL save completes
4. Status bar events `{"status": "ベクトルインデックス構築中: X/Y"}` show progress

---

## Skill: Add UI Lock to a New Streaming Function

**When**: You add a new frontend function that starts an SSE stream and want all action buttons disabled while it runs.

**For GET streams** (`startStream` — returns an EventSource):
```javascript
async function myStreamFn() {
  // Start stream first so we have the EventSource reference for the cancel fn
  const _es = startStream("/api/my/endpoint", outputEl, {
    onDone: () => {
      _unlockUI();
      // ... rest of done handler
    },
    onError: (err) => {
      _unlockUI();
      // ... rest of error handler
    },
  });
  // Lock after stream starts; pass cancel fn so ⏹ 停止 can close EventSource
  _lockUI(() => _es.close());
}
```

**For POST streams** (`startPostStream` — async, returns cancel fn after completion):
```javascript
async function myPostStreamFn() {
  _lockUI(null);   // lock before await; no client-side cancel available
  await startPostStream("/api/my/endpoint", body, outputEl, {
    onDone: () => {
      _unlockUI();
      // ...
    },
    onError: (err) => {
      _unlockUI();
      // ...
    },
  });
  _unlockUI();  // safety net — idempotent if already called by onDone/onError
}
```

**Rules**:
- `_lockUI` is idempotent — calling it twice is safe (second call is a no-op)
- `_unlockUI` restores each button's pre-lock `disabled` state, so buttons that were already disabled (e.g., `generate-report-btn` before an index is built) remain disabled after unlock
- The global **⏹ 停止** button calls `_activeCancel()` (if set), signals `/api/generate/cancel`, then calls `_unlockUI()`

---

## Skill: Debug Streaming / Timeout Issues

**Symptoms**: "stream idle timeout", partial responses, frozen progress bar.

**Checklist**:
1. **Heartbeat**: `_sse_response()` in both route files emits `{"heartbeat": true}` every 15s via a background thread — independent of the LLM generator blocking. Verify both `generation_routes.py` and `explain_routes.py` use the thread-based version.
2. **Client idle timer**: `startStream()` in `stream.js` reconnects after `IDLE_TIMEOUT = 300000` ms (5 min). With thread-based heartbeats this should never fire.
3. **Ollama read timeout**: `_READ_TIMEOUT = 7200` (2h) in `ollama_client.py`. Only increase this if you're using an extremely large model.
4. **Embedding timeout**: `_EMBED_TIMEOUT = 60` in `vector_adapter.py`. Increase if embedding calls are timing out (rare — `nomic-embed-text` is fast).
5. **Large batches**: `analysis_service.py` saves JSONL every `_INCREMENTAL_SAVE_INTERVAL = 50` files. For very slow LLMs, reduce to 20.
6. **Flask buffering**: `X-Accel-Buffering: no` header is set in `_SSE_HEADERS`. If behind nginx, ensure `proxy_buffering off` is set.
7. **UI stuck locked**: If a stream exits without firing `onDone`/`onError`, click **⏹ 停止** to manually call `_unlockUI()`.

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
const _es = startStream("/api/endpoint", null, {
    onMyEvent: (payload) => { /* handle it */ },
    onDone: () => { _unlockUI(); },
    onError: (err) => { _unlockUI(); },
});
_lockUI(() => _es.close());
```

**Note**: The `status` event type (`{"status": "message"}`) is already handled by `_dispatch()` and calls `updateStatusBar(data.status)` — reuse it for any text status update rather than adding a new event type.

---

## Skill: Support a New Thinking Model

**When**: A new Ollama model exposes reasoning in a non-standard way.

**Two patterns already supported**:

1. **`thinking` JSON field** (Gemma, QwQ, etc.):
   - Ollama returns `{"response": "...", "thinking": "..."}` chunks
   - `ollama_client.py` yields thinking text with `\x01` prefix: `yield f"\x01{thinking}"`
   - Route layer (`_sse_response`) detects `\x01`, wraps in `<think>...</think>`, emits as `raw_token` only
   - No changes needed — handled automatically

2. **`<think>` XML tags** (DeepSeek-R1, etc.):
   - Model outputs `<think>...</think>` inline in the `response` field
   - `OllamaPanel.appendToken()` in `stream.js` parses these tags and routes content to the thinking pane
   - No changes needed — handled automatically

**To add a new pattern**:
- If the model uses a different field name (e.g., `"reasoning"`): add a third `yield` block in `ollama_client.py:stream_completion()` following the `thinking` field pattern
- If the model uses different XML tags: update `OllamaPanel.appendToken()` in `stream.js` to detect them

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
