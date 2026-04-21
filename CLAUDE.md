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
│   └── exceptions.py   # Exception hierarchy
│
├── application/         # Business logic — no I/O, no HTTP
│   ├── analysis_service.py    # Incremental indexing, hybrid file reading, LLM batching, parallel RAG embedding
│   ├── context_service.py     # All LLM prompt assembly, token budget management
│   ├── explanation_service.py # Report generation (11 sections), Q&A orchestration
│   ├── generation_service.py  # Plan generation, file generation, git commits
│   ├── project_service.py     # Project management, mode detection, state
│   └── resume_service.py      # Resume mode coordination
│
├── infrastructure/      # Adapters — all I/O lives here
│   ├── ollama_client.py       # HTTP wrapper for Ollama /api/generate (streaming) and /api/embeddings
│   ├── filesystem_adapter.py  # Path operations, file tree building
│   ├── git_adapter.py         # Git operations via GitPython
│   ├── index_adapter.py       # JSONL/JSON persistence for file indexes
│   └── vector_adapter.py      # ChromaDB + Ollama embeddings for RAG semantic search
│
└── interface/           # Flask routes, templates, static assets
    ├── server.py              # App factory, dependency injection
    ├── routes/
    │   ├── project_routes.py  # /api/project/*
    │   ├── generation_routes.py # /api/generate/*
    │   ├── explain_routes.py  # /api/explain/*
    │   └── git_routes.py      # /api/git/*
    ├── templates/
    │   ├── base.html          # Layout + Ollama live panel + global stop button
    │   └── partials/          # plan_viewer, resume_panel, report_viewer
    └── static/
        ├── css/app.css        # Dark theme, no external CSS framework
        └── js/
            ├── stream.js      # SSE + OllamaPanel + heartbeat + reconnection
            ├── app.js         # Main SPA logic, tab switching, UI lock, RAG migration
            ├── chat.js        # Q&A chat UI
            └── filetree.js    # File tree rendering
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
- **Fallback**: If ChromaDB is unavailable, `get_top_chunks_semantic()` falls back to keyword search automatically
- **Search entry point**: `AnalysisService.get_top_chunks_semantic(chunks, query, top_n)` — use this everywhere, never call `get_top_chunks_by_keywords` directly from new code

### Thinking Model Support
Models that emit a `thinking` field in Ollama's JSON response (e.g., Gemma) are handled transparently:
- `ollama_client.py` yields thinking tokens prefixed with `\x01`
- `_sse_response()` in the route layer detects the `\x01` prefix, strips it, and emits `{"raw_token": "<think>...</think>"}` — skipping the main `token` event so thinking text never appears in the main generation output
- The Ollama live panel (`OllamaPanel` in `stream.js`) parses `<think>` tags and routes them to the thinking content area

### Token Budget
`ContextService._guard_budget()` warns (does not truncate) when a prompt exceeds the project's `token_limit` (default 6000). Each project can override via `.localforge/config.json`.

### File Index Structure
- `.localforge/index.jsonl` — per-file summaries (JSONL, incremental)
- `.localforge/project_index.json` — master document with project summary
- `.localforge/chroma/` — ChromaDB vector collection (auto-created by `build_index`)
- `.localforge/config.json` — project config (model, mode, token_limit)
- `.localforge/context.md` — rolling project memory
- `.localforge/generation_log.jsonl` — LLM call history
- `.localforge/report.md` — saved explanation report
- `.localforge/qa_history.md` — Q&A log

## Ollama Lifecycle

`main.py` ensures Ollama is terminated when the app exits, freeing VRAM/RAM:

- **Normal close**: `_kill_ollama()` is called immediately after `webview.start()` returns
- **Python exit**: `atexit.register(_kill_ollama)` covers interpreter shutdown
- **SIGTERM / SIGINT**: signal handlers call `_kill_ollama()` then `os._exit(0)`
- **SIGKILL**: cannot be caught — unavoidable OS limitation

`_kill_ollama()` uses `pkill -x ollama` on Linux/Mac and `taskkill /F /IM ollama.exe` on Windows.

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
- **Embedding phase slow**: Embedding runs 4 workers in parallel after JSONL indexing. Status bar shows `ベクトルインデックス構築中: X/Y`. If `nomic-embed-text` is not pulled, embedding silently falls back to keyword search.
- **ChromaDB errors**: Check `.localforge/chroma/` exists and is writeable; delete it to force full re-embedding on next `build_index()` call (auto-heal will repopulate it).
- **Embedding failures**: Ensure `nomic-embed-text:latest` is pulled in Ollama; app falls back to keyword search if embedding fails.
- **Ollama timeouts**: `_READ_TIMEOUT = 7200` (2h) in `ollama_client.py`. Increase further only for extremely large models.
- **Token budget warnings**: Visible in `app.log`; increase `token_limit` in project config.
- **UI stays locked after error**: If a stream terminates unexpectedly without calling `onDone`/`onError`, click the `⏹ 停止` button to manually unlock the UI.

## Testing

```bash
pytest tests/
```

Tests use mock adapters — no real Ollama or filesystem required. `VectorAdapter` is not injected in test fixtures (`vector=None`), so embedding is skipped and `get_top_chunks_semantic()` falls back to keyword search.
