# LocalForge — Personalized Local AI Code Generation & Project Intelligence IDE

A production-grade, fully local Python desktop application that serves as an AI-powered code generation and project intelligence tool. LocalForge runs entirely on your machine using Ollama as the LLM backend, presenting as a native desktop window (via `pywebview`) backed by a local Flask server.

## Features

### Three Modes

1. **Generate** — Create new projects from scratch using AI-driven planning and file generation
2. **Resume** — Continue and extend any existing project (LocalForge-generated or foreign)
3. **Explain** — Deeply analyze any codebase, produce a structured intelligence report, and answer questions interactively

### RAG-Powered Code Intelligence

LocalForge uses **Retrieval-Augmented Generation (RAG)** for the Explain mode:
- Every file is summarized and embedded into a **ChromaDB** vector store using `nomic-embed-text:latest`
- Q&A and report generation retrieve the most semantically relevant file summaries rather than scanning everything
- Embeddings are built **in parallel** (4 concurrent workers) after the JSONL indexing phase — status bar shows progress
- Incremental: only changed files are re-indexed and re-embedded on subsequent runs
- Fallback: if ChromaDB or the embedding model is unavailable, keyword search is used automatically

### Thinking Model Support

Models that expose a `thinking` field (e.g., Gemma) have their reasoning streamed to the **Ollama live panel** (collapsible right sidebar) without polluting the main generation output. Models using `<think>` XML tags (e.g., DeepSeek-R1) are also supported.

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

## Installation

### Option A — pip (classic)

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

### Option C — Docker (LAN access, headless)

Run LocalForge as a headless Flask server inside a Docker container. The native
desktop window is not available, but the full UI is accessible from **any browser
on the same Wi-Fi network** — useful for doing the heavy Ollama work on a powerful
machine while browsing from a laptop, tablet, or phone.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed on the host
- Ollama **running on the host machine** (the container connects to it so the GPU stays on the host)

#### Build and start

```bash
git clone <repo-url>
cd localforge_web

# Optional: set the directory that holds your projects (default: ~/projects)
export PROJECTS_DIR=~/projects

docker compose up --build
```

The first build takes a few minutes (compiling chromadb native extensions).
Subsequent starts are instant.

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

Because there is no desktop environment inside the container, the native folder
dialog is not available. When you click **📁 Open Folder** the app will show a
text prompt — type the in-container path to your project:

```
/projects/my-app
```

The `~/projects` directory on the host is mounted at `/projects` inside the
container. Set `PROJECTS_DIR` to a different path if your projects live elsewhere:

```bash
PROJECTS_DIR=/home/alice/code docker compose up
```

#### Ollama connectivity

Ollama runs on the host machine and is reached via `host.docker.internal:11434`
(pre-configured). No separate Ollama container is needed and the GPU is used
directly by the host Ollama process.

To point at a different Ollama instance, edit `docker-compose.yml`:

```yaml
environment:
  OLLAMA_HOST: "http://192.168.1.10:11434"
```

#### Restrict to localhost only

If you do not want LAN access, change the port binding in `docker-compose.yml`:

```yaml
ports:
  - "127.0.0.1:7331:7331"
```

#### Stop the container

```bash
docker compose down          # stop; keep persisted index data
docker compose down -v       # stop and delete persisted data
```

---

### Option B — Poetry

#### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:

```bash
poetry --version
```

> On some systems you may need to add `~/.local/bin` to your `PATH`. The installer prints the exact command if needed.

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

After this you can run `python main.py` or `localforge` directly without the `poetry run` prefix.

### Platform-specific notes for pywebview

- **macOS**: No additional dependencies needed (uses WebKit)
- **Linux**: Requires GTK3 + WebKit2GTK:
  ```bash
  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0
  ```
- **Windows**: No additional dependencies needed (uses Edge WebView2)

## Running

### pip

```bash
python main.py
```

### Docker

```bash
docker compose up            # starts and opens http://<host-ip>:7331
docker compose up --build    # rebuild image first (after dependency changes)
docker compose up -d         # run in background
```

### Poetry

```bash
# Using the installed script entry point:
poetry run localforge

# Or the explicit form:
poetry run python main.py

# Or inside an activated shell (poetry shell):
localforge
```

This will:
1. Start a Flask server on `http://127.0.0.1:7331`
2. Open a native desktop window via pywebview
3. On close, automatically terminate the Ollama process to free VRAM/RAM

If pywebview is not available, open your browser to `http://127.0.0.1:7331` manually.

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

## Architecture

```
localforge/
├── domain/           # Models, ports (interfaces), exceptions
├── application/      # Business logic services
├── infrastructure/   # Ollama client, filesystem, git, index, vector adapters
└── interface/        # Flask routes, Jinja2 templates, static assets
```

### Key Design Decisions

- **Clean Architecture**: Strict layer separation — no business logic in routes
- **SSE Streaming**: All LLM output streamed via Server-Sent Events with thread-based heartbeats (every 15s, independent of LLM blocking)
- **Incremental Indexing**: Only re-processes files that have changed (mtime + size check)
- **Parallel RAG Embedding**: `ThreadPoolExecutor(max_workers=4)` embeds chunks into ChromaDB after JSONL indexing; status bar shows `ベクトルインデックス構築中: X/Y`
- **Auto-Heal ChromaDB**: If `.localforge/chroma/` is missing or stale, `build_index()` automatically backfills it from the existing JSONL index
- **UI Lock**: All action buttons are disabled during any active stream; a global stop button (`⏹ 停止`) lets users cancel immediately
- **Token Budget Guard**: Every LLM call checks token budget before execution
- **Hybrid File Reading**: Files >200 lines use structural landmarks (AST for Python, regex for JS/TS)
- **Thinking Model Support**: Models emitting a `thinking` field (Gemma) or `<think>` tags (DeepSeek) have their reasoning routed to the Ollama live panel
- **Ollama Cleanup**: Ollama process is killed on app close via SIGTERM handler, atexit, and post-webview shutdown call
- **Model VRAM Unload on Switch**: When the user selects a different model, the previous model is immediately evicted from VRAM via `keep_alive: 0` before the new model loads — prevents multiple large models accumulating in memory
- **No Default Model**: `ProjectConfig.model` defaults to empty string; every route validates it and returns a clear error before calling Ollama if no model is selected

## `.localforge/` Directory

Each project gets a `.localforge/` metadata directory:

```
.localforge/
├── config.json           # Project settings, selected model
├── context.md            # Rolling project memory
├── project_index.json    # Master project summary document
├── index.jsonl           # Per-file summaries (incremental index)
├── chroma/               # ChromaDB vector collection (RAG embeddings)
├── generation_log.jsonl  # LLM interaction log
├── report.md             # Saved explanation report
├── qa_history.md         # Q&A conversation log
└── app.log               # Rotating application log (not git-tracked)
```

## Running Tests

```bash
# pip
pip install pytest pytest-mock
python -m pytest tests/ -v

# Poetry (dev dependencies are already installed by `poetry install`)
poetry run pytest tests/ -v
```

Tests use mock adapters — no real Ollama, ChromaDB, or filesystem access required.

## Security

- Flask binds to `127.0.0.1` only — never accessible from the network
- File access API validates paths against project root (no path traversal)
- All LLM calls routed through `ollama_client.py` only
- ChromaDB telemetry disabled (`anonymized_telemetry=False`)

## Configuration

Edit `.localforge/config.json` in your project to customize:
- `model`: Ollama model name (default: `llama3.2`)
- `token_limit`: Token budget per LLM call (default: `6000`)
