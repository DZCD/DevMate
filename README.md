# DevMate

AI-powered coding assistant with MCP, RAG, and Skills integration.

## Features

- **Web Search**: Real-time web search via Tavily API through MCP (Model Context Protocol) with Streamable HTTP transport
- **Knowledge Base (RAG)**: Local document indexing and retrieval using ChromaDB vector store
- **Skills System**: Reusable knowledge patterns and code templates stored as markdown files
- **File Operations**: Create, write, and browse files within a sandboxed workspace
- **LangSmith Integration**: Full observability for agent tracing and debugging
- **Modular Architecture**: Clean separation between MCP server, RAG engine, agent core, and CLI

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     CLI (click)                       │
│              devmate init/chat/run/serve              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  DevMate Agent                        │
│         LangChain ReAct Agent + ChatAnthropic         │
├──────────┬───────────┬───────────┬───────────────────┤
│ MCP Tools│  RAG Tool │Skill Tools│  File Tools       │
└────┬─────┴─────┬─────┴─────┬─────┴───────┬───────────┘
     │           │           │             │
┌────▼─────┐ ┌───▼───┐ ┌─────▼────┐ ┌─────▼──────┐
│MCP Server│ │ChromaDB│ │.skills/  │ │ File System│
│(Tavily)  │ │(Vector)│ │(Markdown)│ │ (Workspace)│
│:8001/mcp │ │(Local) │ │          │ │            │
└──────────┘ └───────┘ └──────────┘ └────────────┘
```

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Tavily API key (sign up at [tavily.com](https://tavily.com))
- Anthropic-compatible LLM API access

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/DZCD/DevMate.git
cd DevMate
uv sync
```

### 2. Configure

```bash
cp config.toml.example config.toml
# Edit config.toml and fill in your API keys
```

Required configuration in `config.toml`:

```toml
[model]
base_url = "https://open.bigmodel.cn/api/anthropic"
api_key = "your_api_key_here"
model_name = "glm-5-turbo"

[search]
tavily_api_key = "your_tavily_api_key_here"
```

### 3. Initialize knowledge base

```bash
uv run devmate init
```

### 4. Start MCP server (in a separate terminal)

```bash
uv run devmate serve
```

### 5. Chat with DevMate

```bash
uv run devmate chat
```

Or run a single task:

```bash
uv run devmate run "Build a FastAPI service for managing hiking trails"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `devmate init` | Initialize document index for RAG |
| `devmate chat` | Start interactive chat session |
| `devmate run "prompt"` | Execute a single task |
| `devmate serve` | Start the MCP search server |
| `devmate --version` | Show version |
| `devmate -v chat` | Verbose logging mode |

## Skills System

Skills are markdown files stored in `.skills/` with YAML frontmatter:

```markdown
---
name: "create_fastapi_service"
description: "Create a FastAPI service with standard structure"
trigger_keywords:
  - "fastapi"
  - "api service"
  - "rest api"
---

## Steps
1. Create project structure
2. Add dependencies
3. Implement main module
```

The agent automatically matches skills based on trigger keywords in user queries.

## RAG Knowledge Base

Place markdown documents in the `docs/` directory. They will be:

1. Parsed with `MarkdownHeaderTextSplitter` (respecting headers)
2. Chunked with `RecursiveCharacterTextSplitter` (configurable size/overlap)
3. Stored in ChromaDB for semantic retrieval

Configuration:

```toml
[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"
chunk_size = 1000
chunk_overlap = 200
```

## MCP Server

The MCP server uses **Streamable HTTP** transport (stateless mode) with:

- `mcp.server.lowlevel.Server` as the core MCP implementation
- `StreamableHTTPSessionManager` for HTTP transport
- `Starlette` as the ASGI framework
- Endpoint: `http://localhost:8001/mcp`

## LangSmith Integration

Enable tracing by setting your LangSmith API key in `config.toml`:

```toml
[langsmith]
enabled = true
langchain_api_key = "your_langsmith_api_key"
langchain_project = "devmate"
```

### Verified Trace Example

Successful LangSmith trace captured during real task execution:

- Trace: <https://smith.langchain.com/o/f84fbc14-50a8-44fe-9c85-716ce58215f6/projects/p/3a5251db-8113-439f-9210-4dc44f80828c/r/019d6d56-8abd-72e2-b9de-b294da3b79b5?trace_id=019d6d56-8abd-72e2-b9de-b294da3b79b5&start_time=2026-04-08T13:44:41.406057>
- Task used for verification: `Create a minimal FastAPI hello-world service with one /health endpoint.`

## Docker

### Using Docker Compose

```bash
docker compose up --build
```

This starts:
- **chromadb**: Chroma vector database on port 8000
- **mcp-server**: MCP search server on port 8001
- **devmate**: interactive agent container running `python -m devmate chat`

**Interaction notes:**
- The `devmate` service is configured with `stdin_open: true` and `tty: true`, so it is intended to run in interactive chat mode.
- `docker compose up --build` is suitable for starting the full stack and viewing logs.
- If you want a cleaner direct chat session with the agent, use one of these commands:

```bash
# Start the full stack
docker compose up --build

# In another terminal, open an interactive chat session in the devmate container
docker compose exec devmate python -m devmate chat

# Or start a one-off interactive devmate session
docker compose run --rm devmate chat
```

This makes the expected interaction path explicit for reviewers while keeping `docker compose up --build` as the main startup command.

### Using Dockerfile directly

```bash
docker build -t devmate .
docker run -v ./config.toml:/app/config.toml:ro -p 8001:8001 devmate
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_config.py -v

# Run with verbose logging
uv run pytest tests/ -v -s
```

## Code Quality

```bash
# Format code
uv run ruff format src/ mcp_server/ tests/

# Lint check
uv run ruff check src/ mcp_server/ tests/

# Verify no print() statements
grep -rn "print(" src/ mcp_server/ tests/
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangChain + create_react_agent |
| LLM | Anthropic-compatible (ChatAnthropic) |
| MCP Transport | Streamable HTTP (stateless) |
| Web Search | Tavily API |
| Vector Store | ChromaDB |
| Text Splitting | LangChain Text Splitters |
| Observability | LangSmith |
| CLI | Click |
| ASGI Server | Uvicorn + Starlette |
| Package Manager | uv |
| Linting | Ruff |

## Project Structure

```
DevMate/
├── pyproject.toml              # Project config & dependencies
├── config.toml                 # Runtime config (not committed)
├── config.toml.example         # Config template (committed)
├── Dockerfile                  # Container image
├── docker-compose.yml          # Multi-service orchestration
├── src/devmate/
│   ├── __init__.py             # Package init
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # Configuration loader
│   ├── agent.py                # ReAct agent core
│   ├── rag.py                  # RAG engine (ChromaDB)
│   ├── skills.py               # Skills system
│   └── file_tools.py           # File operation tools
├── mcp_server/
│   ├── __init__.py             # MCP server (Streamable HTTP)
│   └── server.py               # Server entry point
├── docs/                       # RAG documents
├── .skills/                    # Skill markdown files
└── tests/                      # Test suite
```

## License

MIT
