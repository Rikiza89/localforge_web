# LocalForge — Personalized Local AI Code Generation & Project Intelligence IDE

A production-grade, fully local Python desktop application that serves as an AI-powered code generation and project intelligence tool. LocalForge runs entirely on your machine using Ollama as the LLM backend, presenting as a native desktop window (via `pywebview`) backed by a local Flask server on port 7331.

## Features

### Three Modes

| Mode | Description |
|---|---|
| **Generate** | Create new projects from scratch using AI-driven planning and file generation |
| **Resume** | Continue and extend any existing project (LocalForge-generated or foreign) |
| **Explain** | Deeply analyze any codebase, produce a structured intelligence report, and answer questions interactively |

### RAG-Powered Code Intelligence

LocalForge uses **Retrieval-Augmented Generation (RAG)** for the Explain mode:
- Every file is summarized and embedded into a **ChromaDB** vector store using `nomic-embed-text:latest`
- Q&A and report generation retrieve the most semantically relevant file summaries rather than scanning everything
- Embeddings are built **in parallel** (4 concurrent workers) after the JSONL indexing phase — status bar shows progress
- Incremental: only changed files are re-indexed and re-embedded on subsequent runs
- Fallback: if ChromaDB or the embedding model is unavailable, BM25 keyword search is used automatically

### Multi-Layer Caching for Fast Responses

All hot-path data is cached so repeated questions and unchanged files do not add latency:

| Cache | What it stores | Invalidation |
|---|---|---|
| **Response cache** | Full Q&A answers keyed by question + context hash | Index rebuild or question change |
| **File content cache** | Raw file text keyed by path + mtime + size | File modification |
| **Semantic search cache** | Top-N results keyed by query + chunk count | Index rebuild |
| **index_json cache** | Serialized project index JSON | Index mtime change |

Cache hot layer lives in memory; cold layer persists to `.localforge/cache/` so it survives restarts.

### Background Model Warm-Up

At the start of every Q&A and report pipeline, LocalForge fires a background thread that starts loading the Ollama model into RAM in parallel with context assembly (file reads, vector search, prompt building). This means the model is loaded — or much further along — by the time the first Ollama call is made, reducing first-token latency significantly.

### Thinking Model Support

Models that expose a `thinking` field (e.g., Gemma, QwQ) or use `<think>` XML tags (e.g., DeepSeek-R1) have their reasoning streamed to the **Ollama live panel** (collapsible right sidebar) without polluting the main generation output.

---

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.2
```

For RAG-powered Explain mode, also pull the embedding model:

```bash
ollama pull nomic-embed-text:latest
```

Start the Ollama server (runs automatically on most systems):
```bash
ollama serve
```

### 2. Python 3.10+

Ensure you have Python 3.10 or newer installed.

---

## Installation

### Option A — pip

```bash
# Clone the repository
git clone <repo-url>
cd localforge_web

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B — Poetry

#### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify:

```bash
poetry --version
```

> On some systems you may need to add `~/.local/bin` to your `PATH`.

#### 2. Install project dependencies

```bash
git clone <repo-url>
cd localforge_web
poetry install
```

This creates an isolated virtual environment and installs all runtime and dev dependencies automatically.

#### 3. Activate the environment (optional)

```bash
poetry shell
```

After this you can run `python main.py` directly without the `poetry run` prefix.

### Option C — Docker (LAN access, headless)

Run LocalForge as a headless Flask server inside a Docker container. The native desktop window is not available, but the full UI is accessible from **any browser on the same Wi-Fi network** — useful for doing the heavy Ollama work on a powerful machine while browsing from a laptop, tablet, or phone.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed on the host
- Ollama **running on the host machine** (the container connects to it so the GPU stays on the host)

#### Build and start

```bash
git clone <repo-url>
cd localforge_web
docker compose up --build
```

The first build takes a few minutes (compiling chromadb native extensions). Subsequent starts are instant.

#### Access from any device on the same network

Find the host machine's local IP address:

```bash
# macOS / Linux
hostname -I | awk '{print $1}'
```

```powershell
# Windows PowerShell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object InterfaceAlias -ne Loopback).IPAddress
```

Then open `http://<host-ip>:7331` in any browser on the same Wi-Fi.

#### Opening projects

Because there is no desktop environment inside the container, the native folder dialog is not available. When you click **📁 Open Folder** the app will show a text prompt — type the in-container path to your project:

```
/projects/my-app
```

By default a `projects/` folder is created next to `docker-compose.yml` and mounted at `/projects` inside the container. Point it at an existing directory:

```bash
# Linux / macOS
PROJECTS_DIR=/home/alice/code docker compose up

# Windows CMD
set PROJECTS_DIR=C:\Users\alice\code && docker compose up

# Windows PowerShell
$env:PROJECTS_DIR="C:\Users\alice\code"; docker compose up
```

#### Ollama connectivity

Ollama runs on the host machine and is reached via `host.docker.internal:11434` (pre-configured). No separate Ollama container is needed and the GPU is used directly by the host Ollama process.

To point at a different Ollama instance, edit `docker-compose.yml`:

```yaml
environment:
  OLLAMA_HOST: "http://192.168.1.10:11434"
```

#### Restrict to localhost only

```yaml
ports:
  - "127.0.0.1:7331:7331"
```

#### Stop the container

```bash
docker compose down          # stop; keep persisted index and cache data
docker compose down -v       # stop and delete persisted data
```

### Platform-specific notes for pywebview

| Platform | Notes |
|---|---|
| **macOS** | No additional dependencies needed (uses WebKit) |
| **Linux** | Requires GTK3 + WebKit2GTK: `sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0` |
| **Windows** | No additional dependencies needed (uses Edge WebView2) |

---

## Running

### pip

```bash
python main.py
```

### Poetry

```bash
poetry run python main.py
# or inside an activated shell (poetry shell):
python main.py
```

### Docker

```bash
docker compose up            # starts and opens http://<host-ip>:7331
docker compose up --build    # rebuild image first (after dependency changes)
docker compose up -d         # run in background
```

This will:
1. Start a Flask server on `http://127.0.0.1:7331`
2. Open a native desktop window via pywebview
3. On close, automatically terminate the Ollama process to free VRAM/RAM

If pywebview is not available, open your browser to `http://127.0.0.1:7331` manually.

---

## Usage

### Generate Mode

1. Click **📁 Open Folder** to select an empty directory
2. Type your project description in the prompt area
3. Click **Generate Plan** to create an AI-generated project structure
4. Review the plan, edit if needed, then click **Approve & Generate**
5. Watch as LocalForge generates each file with git commits

> During any generation, all action buttons are disabled and a **⏹ 停止** button appears in the header. Click it to stop generation and unlock the UI immediately.

### Resume Mode

Opens automatically when a previously-worked project folder is detected:
- **LocalForge projects**: Continue incomplete file generation or modify the plan
- **Foreign projects**: View the analysis report, continue Q&A, or generate new files

### Explain Mode

Opens automatically when a code-containing folder without `.localforge/` is detected:
1. Click **⚙ インデックス構築** to analyze the codebase (incremental re-indexing supported)
2. The indexing phase runs in two steps: LLM summary generation → parallel RAG embedding
3. Click **レポート生成** to create an 11-section intelligence report
4. Use the **Q&A Chat** to ask questions about the codebase

**Q&A performance notes:**
- Identical questions return a cached answer instantly (no LLM call)
- File content is cached per mtime — unchanged files are never re-read between questions
- The model starts loading into RAM during context assembly so the wait after prompting is reduced

---

## Architecture

```
localforge/
├── domain/           # Pydantic models, port interfaces, exceptions
├── application/      # Business logic services (no I/O, no HTTP)
├── infrastructure/   # Adapters — all I/O lives here
└── interface/        # Flask routes, Jinja2 templates, static assets
```

### Key Design Decisions

| Decision | Detail |
|---|---|
| **Clean Architecture** | Strict layer separation — no business logic in routes |
| **SSE Streaming** | All LLM output streamed via Server-Sent Events with thread-based heartbeats every 15s, independent of LLM blocking |
| **Parallel preprocessing** | File reads, workspace loading, and semantic searches run concurrently via `ThreadPoolExecutor` before the Ollama call |
| **Background model warm-up** | `OllamaClient.preload_model_async()` starts loading the model into RAM while context is assembled |
| **Multi-layer caching** | Response, file content, semantic search, and index_json caches eliminate redundant work across Q&A calls |
| **Async log writes** | `generation_log.jsonl` appends run in background threads — the `done` SSE event is never blocked by disk I/O |
| **Incremental Indexing** | Only re-processes files that have changed (mtime + size check) |
| **Parallel RAG Embedding** | `ThreadPoolExecutor(max_workers=4)` embeds chunks into ChromaDB after JSONL indexing; status bar shows `ベクトルインデックス構築中: X/Y` |
| **Auto-Heal ChromaDB** | If `.localforge/chroma/` is missing or stale, `build_index()` automatically backfills it |
| **UI Lock** | All action buttons disabled during active stream; global **⏹ 停止** button cancels immediately |
| **Token Budget Guard** | Every LLM call checks token budget before execution |
| **Hybrid File Reading** | Files >350 lines use structural landmarks (AST for Python, regex for JS/TS) |
| **Thinking Model Support** | Models emitting a `thinking` field (Gemma) or `<think>` tags (DeepSeek) have reasoning routed to the Ollama live panel |
| **Ollama Cleanup** | Ollama process is killed on app close via SIGTERM handler, atexit, and post-webview shutdown call |
| **Model VRAM Unload on Switch** | Previous model immediately evicted via `keep_alive: 0` when user selects a new model |
| **No Default Model** | `ProjectConfig.model` defaults to `""`; every route validates and returns a clear error if no model is selected |

---

## `.localforge/` Directory

Each project gets a `.localforge/` metadata directory (gitignored):

```
.localforge/
├── config.json           # Project settings (model, token_limit)
├── context.md            # Rolling project memory
├── project_index.json    # Master project summary document
├── index.jsonl           # Per-file summaries (incremental index)
├── chroma/               # ChromaDB vector collection (RAG embeddings)
├── generation_log.jsonl  # LLM interaction log (written asynchronously)
├── report.md             # Saved explanation report
├── qa_history.md         # Q&A conversation log
├── cache/
│   ├── responses/        # Q&A response cache (one JSON file per answer)
│   └── semantic/         # Semantic search result cache
└── app.log               # Rotating application log
```

---

## Running Tests

```bash
# pip
pip install pytest pytest-mock
python -m pytest tests/ -v

# Poetry (dev dependencies already installed by `poetry install`)
poetry run pytest tests/ -v
```

Tests use mock adapters — no real Ollama, ChromaDB, or filesystem access required.

---

## Security

- Flask binds to `127.0.0.1` only in non-Docker mode — never accessible from the network
- File access API validates paths against the project root (no path traversal)
- All LLM calls routed through `ollama_client.py` only
- ChromaDB telemetry disabled (`anonymized_telemetry=False`)

---

## Configuration

Edit `.localforge/config.json` in your project to customize:

| Key | Default | Description |
|---|---|---|
| `model` | `""` | Ollama model name (must be set via the UI selector) |
| `token_limit` | `131072` | Token budget per LLM call |
| `mode` | auto-detected | `generate`, `resume`, or `explain` |
