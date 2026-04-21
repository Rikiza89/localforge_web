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
│   ├── analysis_service.py    # Incremental indexing, hybrid file reading, LLM batching, RAG integration
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
    │   ├── base.html          # Layout + Ollama live panel
    │   └── partials/          # plan_viewer, resume_panel, report_viewer
    └── static/
        ├── css/app.css        # Dark theme, no external CSS framework
        └── js/
            ├── stream.js      # SSE + OllamaPanel + heartbeat + reconnection
            ├── app.js         # Main SPA logic, tab switching, RAG migration
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

### SSE Event Types
| Event | Payload | Description |
|---|---|---|
| `token` | `{"token": "text"}` | LLM output token for main display |
| `raw_token` | `{"raw_token": "text"}` | Same token for Ollama live panel (auto-added by `_sse_response`) |
| `progress` | `{"progress": {"done": N, "total": N, "current_file": "..."}}` | Progress bar update |
| `section` | `{"section": "name"}` | Report section header |
| `file_written` | `{"file_written": "path"}` | File generation complete |
| `heartbeat` | `{"heartbeat": true}` | Keep-alive (auto-added every 15s by `_sse_response`) |
| `done` | `{"done": true}` | Stream complete |
| `error` | `{"error": "message"}` | Error occurred |

### RAG / Vector Search
- **Backend**: ChromaDB (embedded, no server) + `nomic-embed-text:latest` via Ollama `/api/embeddings`
- **Persistence**: `.localforge/chroma/` alongside `index.jsonl`
- **Incremental**: `VectorAdapter.needs_reembedding(chunk)` checks mtime+size before re-embedding
- **Migration**: Already-indexed projects use `GET /api/explain/migrate-vector` (or the "RAG移行" button)
- **Fallback**: If ChromaDB is unavailable, `get_top_chunks_semantic()` falls back to keyword search automatically
- **Search entry point**: `AnalysisService.get_top_chunks_semantic(chunks, query, top_n)` — use this everywhere, never call `get_top_chunks_by_keywords` directly from new code

### Token Budget
`ContextService._guard_budget()` warns (does not truncate) when a prompt exceeds the project's `token_limit` (default 6000). Each project can override via `.localforge/config.json`.

### File Index Structure
- `.localforge/index.jsonl` — per-file summaries (JSONL, incremental)
- `.localforge/project_index.json` — master document with project summary
- `.localforge/chroma/` — ChromaDB vector collection
- `.localforge/config.json` — project config (model, mode, token_limit)
- `.localforge/context.md` — rolling project memory
- `.localforge/generation_log.jsonl` — LLM call history
- `.localforge/report.md` — saved explanation report
- `.localforge/qa_history.md` — Q&A log

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

- **Stream timeouts**: All SSE routes emit heartbeats every 15s; client reconnects after 30s idle
- **ChromaDB errors**: Check `.localforge/chroma/` exists and is writeable; delete it to force full re-migration
- **Embedding failures**: Ensure `nomic-embed-text:latest` is pulled in Ollama; app falls back to keyword search if embedding fails
- **Token budget warnings**: Visible in `app.log`; increase `token_limit` in project config

## Testing

```bash
pytest tests/
```

Tests use mock adapters — no real Ollama or filesystem required.
