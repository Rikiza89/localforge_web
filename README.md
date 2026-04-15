# LocalForge — Personalized Local AI Code Generation & Project Intelligence IDE

A production-grade, fully local Python desktop application that serves as an AI-powered code generation and project intelligence tool. LocalForge runs entirely on your machine using Ollama as the LLM backend, presenting as a native desktop window (via `pywebview`) backed by a local Flask server.

## Features

### Three Modes

1. **Generate** — Create new projects from scratch using AI-driven planning and file generation
2. **Resume** — Continue and extend any existing project (LocalForge-generated or foreign)
3. **Explain** — Deeply analyze any codebase, produce a structured intelligence report, and answer questions interactively

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.2
```

Start the Ollama server (runs automatically on most systems):
```bash
ollama serve
```

### 2. Python 3.10+

Ensure you have Python 3.10 or newer installed.

## Installation

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

### Platform-specific notes for pywebview

- **macOS**: No additional dependencies needed (uses WebKit)
- **Linux**: Requires GTK3 + WebKit2GTK:
  ```bash
  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0
  ```
- **Windows**: No additional dependencies needed (uses Edge WebView2)

## Running

```bash
python main.py
```

This will:
1. Start a Flask server on `http://127.0.0.1:7331`
2. Open a native desktop window via pywebview

If pywebview is not available, open your browser to `http://127.0.0.1:7331` manually.

## Usage

### Generate Mode

1. Click **📁 Open Folder** to select an empty directory
2. Type your project description in the prompt area
3. Click **Generate Plan** to create an AI-generated project structure
4. Review the plan, edit if needed, then click **Approve & Generate**
5. Watch as LocalForge generates each file sequentially with git commits

### Resume Mode

Opens automatically when a previously-worked project folder is detected:
- **LocalForge projects**: Continue incomplete file generation or modify the plan
- **Foreign projects**: View the analysis report, continue Q&A, or generate new files

### Explain Mode

Opens automatically when a code-containing folder without `.localforge/` is detected:
1. Click **Build Index** to analyze the codebase (incremental re-indexing supported)
2. Click **Generate Report** to create an 11-section intelligence report
3. Use the **Q&A Chat** at the bottom to ask questions about the codebase

## Architecture

```
localforge/
├── domain/           # Models, ports (interfaces), exceptions
├── application/      # Business logic services
├── infrastructure/   # Ollama client, filesystem, git, index adapters
└── interface/        # Flask routes, Jinja2 templates, static assets
```

### Key Design Decisions

- **Clean Architecture**: Strict layer separation — no business logic in routes
- **SSE Streaming**: All LLM output streamed via Server-Sent Events
- **Incremental Indexing**: Only re-processes files that have changed (mtime + size check)
- **Token Budget Guard**: Every LLM call checks token budget before execution
- **Parallel Summarization**: `ThreadPoolExecutor(max_workers=3)` for file analysis
- **Hybrid File Reading**: Files >200 lines use structural landmarks (AST for Python)

## `.localforge/` Directory

Each project gets a `.localforge/` metadata directory:

```
.localforge/
├── config.json           # Project settings, selected model
├── context.md            # Rolling project memory
├── project_index.json    # Master project summary document
├── index.jsonl           # Per-file summaries (incremental index)
├── generation_log.jsonl  # LLM interaction log
└── app.log               # Rotating application log (not git-tracked)
```

## Running Tests

```bash
pip install pytest pytest-mock
python -m pytest tests/ -v
```

## Security

- Flask binds to `127.0.0.1` only — never accessible from the network
- File access API validates paths against project root (no path traversal)
- All LLM calls routed through `ollama_client.py` only

## Configuration

Edit `.localforge/config.json` in your project to customize:
- `model`: Ollama model name (default: `llama3.2`)
- `token_limit`: Token budget per LLM call (default: `6000`)
