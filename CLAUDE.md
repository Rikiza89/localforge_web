# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> The project root is `localforge_web/` (this file's directory), one level below the workspace folder `LocalForge_web_best/`. All commands below assume `cd localforge_web` first.

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
│   ├── generation_service.py  # Plan generation, file generation, git commits, resume coordination
│   └── project_service.py     # Project management, mode detection (generate/resume/explain), state
│   (Note: resume mode has no dedicated service — it is coordinated by generation_service + project_service)
│
├── infrastructure/      # Adapters — all I/O lives here
│   ├── ollama_client.py       # HTTP wrapper for Ollama /api/generate (streaming), /api/tags, /api/ps
│   │                          # Includes preload_model_async() and unload_model()
│   ├── disk_cache.py          # Dual-layer cache: in-memory LRU dict + JSON files on disk
│   ├── filesystem_adapter.py  # Path operations, file tree building
│   ├── git_adapter.py         # Git operations via GitPython
│   ├── index_adapter.py       # JSONL/JSON persistence for file indexes
│   ├── vector_adapter.py      # ChromaDB + in-process sentence-transformers (all-MiniLM-L6-v2) embeddings for RAG
│   ├── bm25_adapter.py        # BM25 keyword search fallback (used when ChromaDB unavailable)
│   ├── symbol_extractor.py    # tree-sitter AST symbol extraction (Python/JS/TS); regex fallback for SQL/others
│   ├── code_validator.py      # Post-generation syntax check + auto-rollback (AST for .py, brace-balance for JS/TS)
│   ├── dependency_resolver.py # Resolves import statements → project-relative file paths (cross-file deps)
│   └── document_extractor.py  # Plain-text extraction from PDF/DOCX/XLSX/PPTX (optional deps, empty string on failure)
│
└── interface/           # Flask routes, templates, static assets
    ├── server.py              # App factory, dependency injection
    ├── routes/
    │   ├── _sse_helpers.py      # _sse_response() lives HERE — heartbeated SSE wrapper, raw_token/heartbeat events
    │   ├── project_routes.py    # /api/project/*
    │   ├── generation_routes.py # /api/generate/*
    │   ├── explain_routes.py    # /api/explain/*
    │   ├── git_routes.py        # /api/git/*
    │   └── workspace_routes.py  # /api/workspace/* — multi-project workspace management
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
2. Use `_sse_response(generator)` (defined in `interface/routes/_sse_helpers.py`) for streaming endpoints — it automatically adds `raw_token` and `heartbeat` events
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
- **Backend**: ChromaDB (embedded, no server) + **in-process `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)** for embeddings — embeddings run inside the Python process, **not** via Ollama. Loaded from `models/all-MiniLM-L6-v2/` or `LOCALFORGE_ST_MODEL_PATH` (see `vector_adapter.py`).
- **Persistence**: `.localforge/chroma/` alongside `index.jsonl`
- **Primary path**: `build_index()` embeds all chunks inline — no separate migration step needed for new projects
- **Parallel embedding**: After JSONL indexing completes, all new chunks are embedded concurrently with `ThreadPoolExecutor(max_workers=4)` and progress is reported via `{"status": "ベクトルインデックス構築中: X/Y"}` events
- **Auto-heal**: `build_index()` also checks cached JSONL chunks via `needs_reembedding()` and backfills any missing from ChromaDB — the "RAG移行" button is only needed for projects indexed with a pre-RAG version of the app
- **Incremental**: `VectorAdapter.needs_reembedding(chunk)` checks mtime+size before re-embedding; unchanged files are skipped
- **Fallback**: If ChromaDB or the sentence-transformers model is unavailable, `get_top_chunks_semantic()` falls back to BM25 keyword search automatically
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

## LLM Backends (Ollama / llama.cpp)

The generation backend is **pluggable**. Both backends implement the same `LLMPort`
(`domain/ports.py`), so the application/service layer never changes. The factory
`_build_llm_backend()` in `interface/server.py` selects the adapter from `LLM_BACKEND`.

| `LLM_BACKEND` | Adapter | Endpoint |
|---|---|---|
| `ollama` (default) | `OllamaClient` (`infrastructure/ollama_client.py`) | Ollama `/api/generate` |
| `llamacpp` | `LlamaCppClient` (`infrastructure/llamacpp_client.py`) | llama-server native `/completion` (with `cache_prompt` KV reuse) |

**Embeddings are backend-independent** — always in-process `sentence-transformers`
(`all-MiniLM-L6-v2`). Switching backend affects generation/Q&A/report only.

### Why llama.cpp on a CUDA-less machine
Ollama on Windows cannot offload to an Intel/AMD integrated GPU. The llama.cpp **Vulkan**
build can (`--n-gpu-layers`), so on a CPU-only laptop with an iGPU (e.g. Intel Arc) it is
often meaningfully faster. `LlamaCppClient` uses the native `/completion` endpoint (not
the OpenAI `/v1/chat/completions`) because LocalForge builds full raw prompts itself and
does not use chat templates — `/completion` takes the prompt verbatim and supports
`cache_prompt: true` (KV prefix reuse), which accelerates the 11-section report and Q&A.

**Key difference vs Ollama**: llama-server fixes `n_ctx` at startup (`--ctx-size`), so
per-request `num_ctx` is ignored by `LlamaCppClient`. Start the server with a generous
context (16384–32768) and rely on `cache_prompt`.

### llama-server lifecycle
`LlamaServerManager` (`infrastructure/llamacpp_server.py`) optionally launches and
health-checks `llama-server`. With `LLAMACPP_AUTO_START=1` it spawns the process (Vulkan
offload via `--n-gpu-layers`); otherwise LocalForge attaches to an already-running server
at `LLAMACPP_SERVER_URL`. It is stopped on exit by `main.py` `_stop_llamacpp_server()`
(only a process LocalForge itself started is terminated).

### Backend / CPU environment variables

| Variable | Default | Effect |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `llamacpp` |
| `LLAMACPP_SERVER_URL` | `http://127.0.0.1:8081` | llama-server URL to connect to |
| `LLAMACPP_AUTO_START` | unset | `1` → LocalForge launches llama-server itself |
| `LLAMACPP_BINARY` | `./llamacpp/llama-server[.exe]` | path to the llama-server executable |
| `LLAMACPP_MODEL_PATH` | — | GGUF model to load (required for auto-start) |
| `LLAMACPP_CTX` | `16384` | `--ctx-size` |
| `LLAMACPP_N_GPU_LAYERS` | `0` | `--n-gpu-layers` (Vulkan iGPU offload; `0` = pure CPU) |
| `LLAMACPP_THREADS` | physical cores | `--threads` |
| `LLAMACPP_EXTRA_ARGS` | — | extra llama-server args (space-separated) |
| `LOCALFORGE_NUM_THREAD` | physical cores (CPU-only) | Ollama `num_thread`. On CPU-only machines, the default is the **physical** core count (not logical) to avoid SMT/E-core oversubscription (`recommended_num_thread()`). |
| `LOCALFORGE_MAX_OUTPUT_TOKENS` | `0` (unlimited) | Caps `num_predict` for **generation** to bound runaway CPU loops. Also settable per-project via `ProjectConfig.max_output_tokens`. |
| `LOCALFORGE_KILL_OLLAMA_ON_EXIT` | `0` | `1` → also hard-kill the Ollama process on exit. By default LocalForge only **unloads the model** (it does not start Ollama, so it should not kill it). |

### Recommended local models for CPU-only boxes
Model choice dominates CPU wall-clock. A good speed/quality balance on a ~12-core / 32 GB
machine is a 7B coder model at Q4 (e.g. `qwen2.5-coder:7b` in Ollama, or
`qwen2.5-coder-7b-instruct-q4_k_m.gguf` for llama.cpp); use a 3B for max speed. 32 GB RAM
can hold up to ~14B Q4, but 7B is the sweet spot for interactive use without a GPU.

### Quick start: llama.cpp + Vulkan (iGPU)
```bash
# 1. Download the prebuilt Windows Vulkan release of llama.cpp (ggml-org/llama.cpp
#    releases → llama-*-bin-win-vulkan-x64.zip) and unzip into ./llamacpp/
# 2. Place a GGUF model somewhere, then:
export LLM_BACKEND=llamacpp
export LLAMACPP_AUTO_START=1
export LLAMACPP_MODEL_PATH=/path/to/qwen2.5-coder-7b-instruct-q4_k_m.gguf
export LLAMACPP_N_GPU_LAYERS=999   # offload all layers it can to the Intel Arc iGPU
python main.py
# (or run llama-server yourself and just set LLM_BACKEND + LLAMACPP_SERVER_URL)
```

## Ollama Lifecycle

`main.py` releases LLM resources when the app exits, freeing VRAM/RAM. `_cleanup()` runs
on normal close, `atexit`, and SIGTERM/SIGINT (SIGKILL cannot be caught). It performs, in
order: `_unload_hf_model()` → `_unload_ollama_model()` → `_stop_llamacpp_server()` →
`_kill_ollama()`.

**Default behavior changed**: LocalForge does **not** start Ollama, so it no longer
hard-kills it by default. `_unload_ollama_model()` sends the currently-selected model a
`keep_alive=0` (the actual memory-freeing goal) without terminating a process the user may
be using elsewhere. `_kill_ollama()` (`pkill -x ollama` / `taskkill /F /IM ollama.exe`) is
now **opt-in** via `LOCALFORGE_KILL_OLLAMA_ON_EXIT=1`. The llama.cpp backend stops only the
`llama-server` process LocalForge itself launched (`LlamaServerManager.stop()`).

### Model Switching and VRAM Management

When the user selects a different model via the UI selector, `POST /api/project/model` is called. The route handler:
1. Captures `old_model = project.config.model` before overwriting
2. Calls `project_svc.set_model()` to save the new model to `config.json`
3. Calls `llm.unload_model(old_model)` — sends `keep_alive: 0` to Ollama's `/api/generate` endpoint, which immediately evicts the previous model from VRAM/RAM

`OllamaClient.unload_model()` is fire-and-forget: if Ollama is unavailable or the model was not loaded, the failure is logged as a warning and the switch still completes. The method is also declared in `LLMPort` (domain/ports.py) so any mock in tests can implement it.

**Model default**: `ProjectConfig.model` defaults to `""` (empty string). Every generation route (`stream_plan`, `stream_generation`, `stream_index`, `stream_report`, `regenerate_file`) checks for an empty model and returns a clear error — *"モデルが選択されていません"* — before touching the LLM. The UI selector always syncs the model to the project config before starting any stream.

## Running the App

```bash
pip install -r requirements.txt   # or: poetry install
python main.py
# Opens native window on http://127.0.0.1:7331
```

On Windows, `start_main.bat` activates the bundled `venv/` and runs `main.py`. Dependencies are declared in two places: `requirements.txt` (full pip set, incl. tree-sitter / sentence-transformers / document-parsing libs) and `pyproject.toml` (Poetry; a leaner core set). Keep both in sync when adding a dependency. Docker/headless LAN mode is documented in `README.md`.

Requires Ollama running locally with at least one model pulled (for generation/Q&A).
RAG embeddings need **no** Ollama model — they use the bundled `sentence-transformers`
`all-MiniLM-L6-v2` (auto-detected from `models/all-MiniLM-L6-v2/`, or set
`LOCALFORGE_ST_MODEL_PATH`). On first run, if the model is absent locally it is
downloaded once from HuggingFace Hub, then cached for fully-offline use.

## Common Debugging

- **Stream timeouts**: All SSE routes emit heartbeats every 15s via a background thread independent of the LLM generator. Client idle timer fires after 300s (5 min) — in practice it should never trigger.
- **Embedding phase slow**: Embedding runs 4 workers in parallel after JSONL indexing. Status bar shows `ベクトルインデックス構築中: X/Y`. Embeddings use in-process `sentence-transformers` (`all-MiniLM-L6-v2`); if that model or ChromaDB is unavailable, search silently falls back to BM25 keyword search.
- **ChromaDB errors**: Check `.localforge/chroma/` exists and is writeable; delete it to force full re-embedding on next `build_index()` call (auto-heal will repopulate it).
- **Semantic cache stale**: Delete `.localforge/cache/semantic/` and restart; it is rebuilt automatically on the next Q&A.
- **Response cache stale**: Delete `.localforge/cache/responses/` or trigger a `build_index()` run (which updates `project_index.json` mtime, making all old cache keys miss automatically).
- **Ollama timeouts**: `_READ_TIMEOUT = 7200` (2h) in `ollama_client.py`. Increase further only for extremely large models.
- **Token budget warnings**: Visible in `app.log`; increase `token_limit` in project config.
- **UI stays locked after error**: If a stream terminates unexpectedly without calling `onDone`/`onError`, click the `⏹ 停止` button to manually unlock the UI.
- **Model loading slow (first Q&A)**: Expected — `preload_model_async()` fires at the start of the pipeline but CPU cold-start can take several minutes. Subsequent Q&As reuse the loaded model via `keep_alive="2h"`.

## Testing

```bash
python -m pytest tests/ -v          # full suite
python -m pytest tests/test_flask_routes.py -v          # one file
python -m pytest tests/test_application_services.py::test_name -v   # one test
python -m pytest tests/ -k explain  # tests matching a keyword
```

Test files are organized by layer: `test_domain_models.py`, `test_application_services.py`, `test_infrastructure.py`, `test_flask_routes.py`, plus integration suites `test_integration_{generate,resume,explain}.py`. Shared fixtures live in `tests/conftest.py`.

Tests use mock adapters — no real Ollama or filesystem required. `VectorAdapter` is not injected in test fixtures (`vector=None`), so embedding is skipped and `get_top_chunks_semantic()` falls back to BM25 keyword search.

## Security Model

LocalForge is designed as a **fully local, offline-first** tool. No data leaves the machine by default. This section documents the threat model, the controls in place, and the environment variables that can change the security posture.

### What stays local

| Component | Network target | Notes |
|---|---|---|
| LLM inference | `http://localhost:11434` (Ollama) | All prompts, file contents, and responses stay on-machine |
| Vector embeddings | None (in-process) | `sentence-transformers` `all-MiniLM-L6-v2` runs inside the Python process — no network call after the one-time first-run model download |
| ChromaDB | Embedded, no network | Telemetry explicitly disabled: `Settings(anonymized_telemetry=False)` |
| Flask server | `127.0.0.1:7331` by default | Binds to loopback only; not reachable from the network |
| All JS assets | Served from `static/` | No external CDN dependencies anywhere |

### Environment variables that affect security posture

These two variables change where data goes. Both default to safe values. The app logs a **startup warning** if either is set to a non-localhost value.

| Variable | Default | Effect if changed |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Redirects all LLM calls (including indexed file contents and prompts) to the specified host |
| `FLASK_HOST` | `127.0.0.1` | If set to `0.0.0.0`, the Flask API is reachable from the local network with no authentication |

**Do not set `OLLAMA_HOST` to an external URL** unless you own and trust that server. All file content that gets indexed and every LLM prompt is sent there.

**Do not set `FLASK_HOST` to `0.0.0.0`** in untrusted network environments. There is no authentication on the API — anyone on the LAN can read project files and trigger generation.

### Input validation and injection controls

- **Path traversal**: All file read/write endpoints use `path.resolve().relative_to(project_root.resolve())` and return 403 if the resolved path escapes the project directory. This covers symlink and `../` attacks.
- **Command injection**: All `subprocess.run()` calls (git operations, nvidia-smi, ollama/pkill) use list arguments — never `shell=True`. User-supplied strings are passed as argument values, not shell tokens.
- **XSS**: User-visible strings are passed through `escapeHtml()` before `innerHTML` assignment. Markdown from LLM output is rendered via `marked.js` and then sanitized by **DOMPurify 3.4.5** (bundled locally in `static/js/purify.min.js`) before being set as `innerHTML`.
- **Flask debug**: `debug=False` is hardcoded in `main.py` — no environment variable override exists.
- **SECRET_KEY**: Set to `os.urandom(32)` at startup (fresh each run). Sessions are not used by this app; the key is set for defense-in-depth.

### First-run network access (sentence-transformers)

On the very first run, if the `all-MiniLM-L6-v2` model is not present locally, `sentence-transformers` downloads it from HuggingFace Hub (`https://huggingface.co`). After download it is cached in `./models/all-MiniLM-L6-v2/` and all subsequent runs are fully offline. To pre-cache it and avoid any outbound call:

```bash
# Before first run, place the model manually:
mkdir -p models/all-MiniLM-L6-v2
# copy model files here, or set:
export LOCALFORGE_ST_MODEL_PATH=/path/to/cached/model
```

### What was audited and found clean

A full security audit (May 2026) confirmed: no external HTTP calls in production code paths, no CORS headers, no `eval()`/`Function()` in JS, no sensitive data in log files, no hardcoded credentials, no shell-injection risk in subprocess calls, and no `{{ var | safe }}` in Jinja templates.
