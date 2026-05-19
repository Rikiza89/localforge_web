# LocalForge — Implementation Skills Reference

Repeatable patterns for common extension and maintenance tasks.
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
1. `application/explanation_service.py` — add the section name to `REPORT_SECTIONS` list
2. `application/context_service.py` — optionally add a specialised `build_report_section_prompt` variant if the section needs a unique prompt structure

**Key list** (in `explanation_service.py`):
```python
REPORT_SECTIONS = [
    "Project Overview",
    "Module Map",
    # ... add new section name here
]
```

The report loop in `stream_report()` iterates this list and pre-fetches all section semantic searches in parallel before the loop starts — adding a new section name is the only change required.

---

## Skill: Add a New Infrastructure Adapter

**When**: You need a new external integration (database, cloud API, etc.).

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
app.config["my_adapter"] = my_adapter
```
4. Add parameter to the service `__init__` that needs it, store as `self._my = my_adapter`

---

## Skill: Use the DiskCache

**When**: You need a persistent cache for expensive computed values (API results, embeddings, rendered output).

**Import**:
```python
from localforge.infrastructure.disk_cache import DiskCache
```

**Usage pattern**:
```python
# Create (typically in __init__)
self._cache = DiskCache(root / ".localforge" / "cache" / "my_cache", max_memory=100)

# Read-through pattern
def get_value(self, key: str) -> str:
    cached = self._cache.get(key)
    if cached is not None:
        return cached
    result = expensive_operation(key)
    self._cache.set(key, result)
    return result

# Values must be plain strings — JSON-encode complex objects
import json
self._cache.set(key, json.dumps(my_dict))
value = json.loads(self._cache.get(key))

# Clear on index rebuild
self._cache.clear()
```

**Notes**:
- Hot layer: in-memory LRU dict (bounded by `max_memory`)
- Cold layer: one JSON file per entry in `cache_dir/<sha256>.json`
- Thread-safe for reads; last-writer-wins for concurrent writes (harmless — same value)
- Call `.clear()` whenever the underlying data changes to prevent stale reads

---

## Skill: Read Files Using the Content Cache

**When**: You need to read a file's content inside `ExplanationService` (Phase A pinned reads, Phase F context reads, or any new pipeline step).

**Always use the cache helper instead of `Path.read_text()`**:
```python
# Good — uses mtime+size cache, never re-reads unchanged files
content = self._read_file_cached(root / "some/file.py", max_chars=3000)
if content is None:
    # file doesn't exist or couldn't be read
    ...

# Full file, no truncation
content = self._read_file_cached(root / "some/file.py", max_chars=-1)

# Bad — bypasses cache, re-reads from disk every call
content = (root / "some/file.py").read_text()
```

The cache key is `(path_str, mtime_ns, size_bytes, max_chars)`. Any modification to the file is automatically a cache miss.

---

## Skill: Manage Ollama Model Switching and VRAM

**When**: You need to understand or change how LocalForge frees memory when the user changes the active model.

**How it works now**:
- `POST /api/project/model` (in `project_routes.py`) captures `old_model` before saving the new one
- Calls `llm.unload_model(old_model)` → `OllamaClient.unload_model()` sends `{"model": old_model, "prompt": "", "keep_alive": 0}` to Ollama `/api/generate`
- Ollama immediately evicts the model from VRAM/RAM on receipt of `keep_alive: 0`
- Failure is logged as a warning only — the model switch always completes regardless

**Background warm-up** (`OllamaClient.preload_model_async`):
- Called at the start of `stream_answer()` and `stream_report()` before any preprocessing
- Sends an empty generate request in a daemon thread using a fresh `requests.Session`
- Ollama loads the model while preprocessing runs in parallel
- Fire-and-forget: errors are silent (DEBUG log only)

**To force-unload a model programmatically**:
```python
llm = current_app.config["llm"]
llm.unload_model("some-large-model")
```

**Where model is stored**: `ProjectConfig.model` in `.localforge/config.json`. Defaults to `""` (no default model). All generation routes return a `400` error if the model is empty.

---

## Skill: Modify LLM Prompts

**When**: You want to change what LocalForge sends to Ollama for any operation.

**Single source of truth**: All prompts live in **`application/context_service.py`** — one method per operation.

| Method | Operation |
|---|---|
| `build_plan_prompt()` | Generate-mode: project plan from user description |
| `build_file_generation_prompt()` | Generate-mode: per-file code generation |
| `build_file_summary_prompt()` | Explain-mode: file summary during indexing |
| `build_qa_prompt()` | Explain-mode: Q&A with RAG context |
| `build_report_section_prompt()` | Explain-mode: one section of the 11-part report |

**To change a prompt**: Edit the corresponding method in `context_service.py`. The calling service passes the return value directly to `self._llm.stream_completion(model, prompt)` — no other changes needed.

**To add a new prompt**: Add a new method following the naming pattern `build_<operation>_prompt()`, then call it from the relevant application service before the `stream_completion` call.

**Token budget notes**: `build_qa_prompt()` uses O(n) history trimming — token costs are pre-computed per message before the trimming loop, not recalculated each iteration.

---

## Skill: Extend the RAG Vector Index

**When**: You want to change embedding model, vector DB, or search strategy.

**Key files**:
- `infrastructure/vector_adapter.py` — all ChromaDB and Ollama embedding logic
- `infrastructure/bm25_adapter.py` — BM25 keyword fallback search
- `application/analysis_service.py` — collects chunks in `embed_queue`, then embeds in parallel with `ThreadPoolExecutor(max_workers=4)` after JSONL indexing completes
- `application/analysis_service.py:get_top_chunks_semantic()` — single entry point for all semantic search (with in-memory + disk cache)

**To change embedding model**:
- Update `_EMBED_MODEL` constant in `vector_adapter.py`
- Delete `.localforge/chroma/` and `.localforge/cache/semantic/` in existing projects to force re-embedding and clear cached results

**To change embedding parallelism**:
- Update `max_workers=4` in the `ThreadPoolExecutor` call inside `build_index()` in `analysis_service.py`

**To add metadata filters** (e.g., search only Python files):
- Update `VectorAdapter.get_top_chunks_semantic()` to pass `where={"language": "python"}` to `self._collection.query()`

**Semantic search cache**: `get_top_chunks_semantic()` caches results in `_semantic_cache` (in-memory) and `.localforge/cache/semantic/` (disk). The cache key is `(query_normalized, top_n, chunk_count)`. It is automatically cleared by `invalidate_index_cache()` at the end of every `build_index()` run.

---

## Skill: Add UI Lock to a New Streaming Function

**When**: You add a new frontend function that starts an SSE stream and want all action buttons disabled while it runs.

**For GET streams** (`startStream` — returns an EventSource):
```javascript
async function myStreamFn() {
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
  _lockUI(() => _es.close());
}
```

**For POST streams** (`startPostStream` — async):
```javascript
async function myPostStreamFn() {
  _lockUI(null);
  await startPostStream("/api/my/endpoint", body, outputEl, {
    onDone: () => { _unlockUI(); },
    onError: (err) => { _unlockUI(); },
  });
  _unlockUI();  // safety net — idempotent
}
```

**Rules**:
- `_lockUI` is idempotent — calling it twice is safe
- `_unlockUI` restores each button's pre-lock `disabled` state
- The global **⏹ 停止** button calls `_activeCancel()` (if set), signals `/api/generate/cancel`, then calls `_unlockUI()`

---

## Skill: Debug Streaming / Timeout Issues

**Symptoms**: "stream idle timeout", partial responses, frozen progress bar.

**Checklist**:
1. **Heartbeat**: `_sse_response()` emits `{"heartbeat": true}` every 15s via a background thread — independent of LLM blocking. Verify the route uses the thread-based version.
2. **Client idle timer**: `startStream()` in `stream.js` reconnects after `IDLE_TIMEOUT = 300000` ms (5 min). With thread-based heartbeats this should never fire.
3. **Ollama read timeout**: `_READ_TIMEOUT = 7200` (2h) in `ollama_client.py`.
4. **Embedding timeout**: `_EMBED_TIMEOUT = 60` in `vector_adapter.py`. Increase if embedding calls are timing out.
5. **Large batches**: `analysis_service.py` saves JSONL every `_INCREMENTAL_SAVE_INTERVAL = 50` files.
6. **Flask buffering**: `X-Accel-Buffering: no` header is set in `_SSE_HEADERS`. If behind nginx, ensure `proxy_buffering off`.
7. **UI stuck locked**: Click **⏹ 停止** to manually call `_unlockUI()`.

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

**Note**: Reuse the existing `status` event (`{"status": "message"}`) for any plain-text status bar update — it is already handled by `_dispatch()` and calls `updateStatusBar(data.status)`.

---

## Skill: Support a New Thinking Model

**When**: A new Ollama model exposes reasoning in a non-standard way.

**Two patterns already supported**:

1. **`thinking` JSON field** (Gemma, QwQ, etc.):
   - Ollama returns `{"response": "...", "thinking": "..."}` chunks
   - `ollama_client.py` yields thinking text with `\x01` prefix
   - Route layer detects `\x01`, wraps in `<think>...</think>`, emits as `raw_token` only
   - No changes needed — handled automatically

2. **`<think>` XML tags** (DeepSeek-R1, etc.):
   - Model outputs `<think>...</think>` inline in the `response` field
   - `OllamaPanel.appendToken()` in `stream.js` parses these tags and routes content to the thinking pane
   - No changes needed — handled automatically

**To add a new pattern**:
- If the model uses a different field name (e.g., `"reasoning"`): add a third `yield` block in `ollama_client.py:stream_completion()` following the `thinking` field pattern
- If the model uses different XML tags: update `OllamaPanel.appendToken()` in `stream.js`

---

## Skill: Tune Token Budget

**When**: LLM responses are getting cut off or prompts are too large.

**Per-project**: Edit `.localforge/config.json`:
```json
{"token_limit": 32768}
```

**Global default** (`application/context_service.py`):
```python
_DEFAULT_TOKEN_LIMIT = 131072  # change here
```

**CPU num_ctx** (controls Ollama KV cache size — smaller = faster prefill):
- Q&A: `_CPU_NUM_CTX = 8192` in `explanation_service.py:stream_answer()`
- Report sections: `_r_num_ctx = 4096` in `explanation_service.py:stream_report()`
- Changing this value causes Ollama to reload the model — keep it fixed to avoid reloads between calls

**Token estimation**: `words × 1.3 ≈ tokens` (defined in `context_service._estimate_tokens()`). History trimming uses O(n) per-message pre-computation — do not revert to the O(n²) loop that rebuilt the full joined string each iteration.
