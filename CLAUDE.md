# LocalForge Web — AI Assistant Orientation

## What This Project Is

LocalForge is a **fully local AI-powered code generation and project intelligence tool**. It runs as a native desktop window (via pywebview) backed by a local Flask server on port 7331. All LLM inference uses **Ollama** running locally — no cloud APIs.

Three modes:
- **Generate**: Create new projects from scratch with AI-driven planning
- **Resume**: Continue or extend any existing project (LocalForge-generated or foreign)
- **Explain**: Deep-analyse any codebase, produce an 11-section intelligence report, enable interactive Q&A

## Architecture (Clean Architecture)

```
localforge/
├── domain/              # Pydantic models, port interfaces, exceptions
│   ├── models.py        # All data models (FileChunk, GenerationPlan, ProjectIndex, ...)
│   ├── ports.py         # Protocol interfaces (LLMPort, FileSystemPort, GitPort, IndexPort, VectorIndexPort)
│   └── exceptions.py    # Exception hierarchy
│
├── application/         # Business logic — no I/O, no HTTP
│   ├── analysis_service.py    # Incremental indexing, hybrid file reading, LLM batching, parallel RAG embedding, semantic search cache
│   ├── context_service.py     # All LLM prompt assembly, token budget management, O(n) history trimming
│   ├── explanation_service.py # Report generation (11 sections), Q&A orchestration, response/file/index_json caches
│   ├── generation_service.py  # Plan generation, file generation, git commits
│   ├── project_service.py     # Project management, mode detection, state
│   └── resume_service.py      # Resume mode coordination
│
├── infrastructure/      # Adapters — all I/O lives here
│   ├── ollama_client.py       # HTTP wrapper for Ollama /api/generate (streaming) and /api/embeddings
│   │                          # Includes preload_model_async() and unload_model()
│   ├── disk_cache.py          # Dual-layer cache: in-memory LRU dict + JSON files on disk
│   ├── filesystem_adapter.py  # Path operations, file tree building
│   ├── git_adapter.py         # Git operations via GitPython
│   ├── index_adapter.py       # JSONL/JSON persistence for file indexes
│   ├── vector_adapter.py      # ChromaDB + Ollama embeddings for RAG semantic search
│   └── bm25_adapter.py        # BM25 keyword search fallback (used when ChromaDB unavailable)
│
└── interface/           # Flask routes, templates, static assets
    ├── server.py              # App factory, dependency injection
    ├── routes/
    │   ├── project_routes.py    # /api/project/*
    │   ├── generation_routes.py # /api/generate/*
    │   ├── explain_routes.py    # /api/explain/*
    │   └── git_routes.py        # /api/git/*
    ├── templates/
    │   ├── base.html            # Layout + Ollama live panel + global stop button
    │   └── partials/            # plan_viewer, resume_panel, report_viewer
    └── static/
        ├── css/app.css          # Dark theme, no external CSS framework
        └── js/
            ├── stream.js        # SSE + OllamaPanel + heartbeat + reconnection
            ├── app.js           # Main SPA logic, tab switching, UI lock, RAG migration
            ├── chat.js          # Q&A chat UI
            └── filetree.js      # File tree rendering
```

## Key Conventions

### Adding a New API Endpoint
1. Add route function to the appropriate `interface/routes/*.py` blueprint
2. Use `_sse_response(generator)` for streaming endpoints — it automatically adds `raw_token` and `heartbeat` events
3. Register nothing in `server.py` — blueprints are already registered

### Adding a New LLM Operation
1. Add a prompt builder to `application/context_service.py` (one method per operation type)
2. Add the business logic method to the appropriate application service
3. Call `self._llm.stream_completion(model, prompt)` and yield `{"token": t}` for each token
4. Yield `{"done": True}` at the end

### Where All Prompts Live

Every prompt sent to Ollama is built in **`application/context_service.py`** — one method per operation:

| Method | Used by |
|---|---|
| `build_plan_prompt()` | `GenerationService.stream_plan()` — project plan from user description |
| `build_file_generation_prompt()` | `GenerationService.stream_all_files()` — per-file code generation |
| `build_file_summary_prompt()` | `AnalysisService.build_index()` — summarise a single file during indexing |
| `build_qa_prompt()` | `ExplanationService.stream_answer()` — Q&A with RAG context |
| `build_report_section_prompt()` | `ExplanationService.stream_report()` — one section of the 11-part report |

To modify any LLM behaviour, edit only `context_service.py`. No prompts exist anywhere else.

### Adding a New Infrastructure Adapter
1. Define a `Protocol` in `domain/ports.py`
2. Implement it in `infrastructure/`
3. Instantiate it in `interface/server.py` `create_app()` and inject into services

### Adding a New Streaming Function in the Frontend
Every function that starts a stream must call `_lockUI` / `_unlockUI` so the UI is disabled during generation:

```javascript
// For GET streams (startStream returns EventSource):
async function myStreamFn() {
  const _es = startStream("/api/my/endpoint", outputEl, {
    onDone: () => { _unlockUI(); /* ... */ },
    onError: (err) => { _unlockUI(); /* ... */ },
  });
  _lockUI(() => _es.close());   // pass cancel fn so stop button can close EventSource
}

// For POST streams (startPostStream):
async function myPostStreamFn() {
  _lockUI(null);                // no client-side cancel; server-side /api/generate/cancel handles it
  await startPostStream("/api/my/endpoint", body, outputEl, {
    onDone: () => { _unlockUI(); /* ... */ },
    onError: (err) => { _unlockUI(); /* ... */ },
  });
  _unlockUI();                  // safety net if onDone/onError is not called
}
```

The global `⏹ 停止` button in the header calls `_activeCancel()` (closes EventSource), signals `/api/generate/cancel`, then `_unlockUI()`.

### SSE Event Types
| Event | Payload | Description |
|---|---|---|
| `token` | `{"token": "text"}` | LLM output token for main display |
| `raw_token` | `{"raw_token": "text"}` | Same token for Ollama live panel (auto-added by `_sse_response`) |
| `progress` | `{"progress": {"done": N, "total": N, "current_file": "..."}}` | Progress bar update |
| `section` | `{"section": "name"}` | Report section header |
| `file_written` | `{"file_written": "path"}` | File generation complete |
| `status` | `{"status": "message"}` | Status bar text update (used by embedding phase) |
| `heartbeat` | `{"heartbeat": true}` | Keep-alive (auto-added every 15s by `_sse_response`) |
| `done` | `{"done": true}` | Stream complete |
| `error` | `{"error": "message"}` | Error occurred |

### RAG / Vector Search
- **Backend**: ChromaDB (embedded, no server) + `nomic-embed-text:latest` via Ollama `/api/embeddings`
- **Persistence**: `.localforge/chroma/` alongside `index.jsonl`
- **Primary path**: `build_index()` embeds all chunks inline — no separate migration step needed for new projects
- **Parallel embedding**: After JSONL indexing completes, all new chunks are embedded concurrently with `ThreadPoolExecutor(max_workers=4)` and progress is reported via `{"status": "ベクトルインデックス構築中: X/Y"}` events
- **Auto-heal**: `build_index()` also checks cached JSONL chunks via `needs_reembedding()` and backfills any missing from ChromaDB — the "RAG移行" button is only needed for projects indexed with a pre-RAG version of the app
- **Incremental**: `VectorAdapter.needs_reembedding(chunk)` checks mtime+size before re-embedding; unchanged files are skipped
- **Fallback**: If ChromaDB is unavailable, `get_top_chunks_semantic()` falls back to BM25 keyword search automatically
- **Search entry point**: `AnalysisService.get_top_chunks_semantic(chunks, query, top_n)` — use this everywhere. It checks in-memory cache → disk cache → actual search before calling the vector DB or BM25.
- **Search cache**: Results are cached in `AnalysisService._semantic_cache` (in-memory) and `.localforge/cache/semantic/` (disk). Both are cleared on every `build_index()` call via `invalidate_index_cache()`.

### Multi-Layer Caching

All hot-path data in `ExplanationService` and `AnalysisService` is cached. **Never bypass these helpers to call raw I/O directly.**

| Cache | Where | Helper | Key |
|---|---|---|---|
| Q&A response | `ExplanationService._response_cache` + disk | `_get_response_disk_cache()` | SHA-256 of root+question+pinned_paths+history tail+index_mtime |
| File content | `ExplanationService._file_content_cache` | `_read_file_cached(path, max_chars)` | `(path, mtime_ns, size, max_chars)` |
| index_json string | `ExplanationService._index_json_cache` | `_get_index_json(root, project_index, include_files)` | `(root_str, mtime, include_files)` |
| Semantic search results | `AnalysisService._semantic_cache` + disk | `get_top_chunks_semantic()` | `(query_norm, top_n, chunk_count)` |

**`DiskCache`** (`infrastructure/disk_cache.py`): shared utility used by the response and semantic caches. Stores one JSON file per entry under a configurable directory. LRU eviction in memory (configurable `max_memory` entries). Call `.clear()` on index rebuild.

**Response cache invalidation**: The cache key includes `index_mtime`, so any `build_index()` run that updates `project_index.json` automatically invalidates stale entries — no explicit wiring required.

**File content cache**: `_read_file_cached(path, max_chars=-1)` where `max_chars=-1` means full file (no truncation). Uses `stat()` mtime_ns + size as the key — any file modification is an automatic cache miss.

### Background Model Warm-Up

`OllamaClient.preload_model_async(model)` fires a daemon thread that sends an empty `POST /api/generate` request to Ollama, which loads the model into RAM without generating tokens. This runs at the **very start** of both `stream_answer()` and `stream_report()`, in parallel with all context assembly phases.

- The method uses a **fresh `requests.Session`** so the main session is not blocked
- It is fire-and-forget: errors are logged at DEBUG level only, never raised
- By the time context assembly finishes (file reads, vector search, prompt building), the model is loaded or much further along — reducing the first-token wait by up to the full preprocessing time

### Thinking Model Support
Models that emit a `thinking` field in Ollama's JSON response (e.g., Gemma) are handled transparently:
- `ollama_client.py` yields thinking tokens prefixed with `\x01`
- `_sse_response()` in the route layer detects the `\x01` prefix, strips it, and emits `{"raw_token": "<think>...</think>"}` — skipping the main `token` event so thinking text never appears in the main generation output
- The Ollama live panel (`OllamaPanel` in `stream.js`) parses `<think>` tags and routes them to the thinking content area

### Token Budget
`ContextService._guard_budget()` warns (does not truncate) when a prompt exceeds the project's `token_limit` (default `131072`). Each project can override via `.localforge/config.json`.

History trimming in `build_qa_prompt()` uses **O(n) per-message pre-computation** — token costs are calculated once per message, not by rebuilding the full joined string each loop iteration.

### File Index Structure
- `.localforge/index.jsonl` — per-file summaries (JSONL, incremental)
- `.localforge/project_index.json` — master document with project summary
- `.localforge/chroma/` — ChromaDB vector collection (auto-created by `build_index`)
- `.localforge/config.json` — project config (model, mode, token_limit)
- `.localforge/context.md` — rolling project memory
- `.localforge/generation_log.jsonl` — LLM call history (written asynchronously via `_log_async()`)
- `.localforge/report.md` — saved explanation report
- `.localforge/qa_history.md` — Q&A log
- `.localforge/cache/responses/` — Q&A response cache (DiskCache, one file per entry)
- `.localforge/cache/semantic/` — semantic search result cache (DiskCache)

## Ollama Lifecycle

`main.py` ensures Ollama is terminated when the app exits, freeing VRAM/RAM:

- **Normal close**: `_kill_ollama()` is called immediately after `webview.start()` returns
- **Python exit**: `atexit.register(_kill_ollama)` covers interpreter shutdown
- **SIGTERM / SIGINT**: signal handlers call `_kill_ollama()` then `os._exit(0)`
- **SIGKILL**: cannot be caught — unavoidable OS limitation

`_kill_ollama()` uses `pkill -x ollama` on Linux/Mac and `taskkill /F /IM ollama.exe` on Windows.

### Model Switching and VRAM Management

When the user selects a different model via the UI selector, `POST /api/project/model` is called. The route handler:
1. Captures `old_model = project.config.model` before overwriting
2. Calls `project_svc.set_model()` to save the new model to `config.json`
3. Calls `llm.unload_model(old_model)` — sends `keep_alive: 0` to Ollama's `/api/generate` endpoint, which immediately evicts the previous model from VRAM/RAM

`OllamaClient.unload_model()` is fire-and-forget: if Ollama is unavailable or the model was not loaded, the failure is logged as a warning and the switch still completes. The method is also declared in `LLMPort` (domain/ports.py) so any mock in tests can implement it.

**Model default**: `ProjectConfig.model` defaults to `""` (empty string). Every generation route (`stream_plan`, `stream_generation`, `stream_index`, `stream_report`, `regenerate_file`) checks for an empty model and returns a clear error — *"モデルが選択されていません"* — before touching the LLM. The UI selector always syncs the model to the project config before starting any stream.

## Running the App

```bash
pip install -r requirements.txt
python main.py
# Opens native window on http://127.0.0.1:7331
```

Requires Ollama running locally with at least one model pulled. For RAG:
```bash
ollama pull nomic-embed-text:latest
```

## Common Debugging

- **Stream timeouts**: All SSE routes emit heartbeats every 15s via a background thread independent of the LLM generator. Client idle timer fires after 300s (5 min) — in practice it should never trigger.
- **Embedding phase slow**: Embedding runs 4 workers in parallel after JSONL indexing. Status bar shows `ベクトルインデックス構築中: X/Y`. If `nomic-embed-text` is not pulled, embedding silently falls back to BM25 keyword search.
- **ChromaDB errors**: Check `.localforge/chroma/` exists and is writeable; delete it to force full re-embedding on next `build_index()` call (auto-heal will repopulate it).
- **Semantic cache stale**: Delete `.localforge/cache/semantic/` and restart; it is rebuilt automatically on the next Q&A.
- **Response cache stale**: Delete `.localforge/cache/responses/` or trigger a `build_index()` run (which updates `project_index.json` mtime, making all old cache keys miss automatically).
- **Ollama timeouts**: `_READ_TIMEOUT = 7200` (2h) in `ollama_client.py`. Increase further only for extremely large models.
- **Token budget warnings**: Visible in `app.log`; increase `token_limit` in project config.
- **UI stays locked after error**: If a stream terminates unexpectedly without calling `onDone`/`onError`, click the `⏹ 停止` button to manually unlock the UI.
- **Model loading slow (first Q&A)**: Expected — `preload_model_async()` fires at the start of the pipeline but CPU cold-start can take several minutes. Subsequent Q&As reuse the loaded model via `keep_alive="2h"`.

## Testing

```bash
pytest tests/
```

Tests use mock adapters — no real Ollama or filesystem required. `VectorAdapter` is not injected in test fixtures (`vector=None`), so embedding is skipped and `get_top_chunks_semantic()` falls back to BM25 keyword search.
