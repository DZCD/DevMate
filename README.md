# DevMate

[中文说明](./README.zh-CN.md)

An AI-powered coding assistant with MCP, RAG, and Skills integration.

## Core Capabilities

| Capability | Description |
|------------|-------------|
| **Web Search** | Real-time web search via Tavily through MCP |
| **Knowledge Retrieval** | RAG system based on ChromaDB, letting AI understand your documents |
| **Skills System** | Reusable prompt templates for standardizing common dev tasks |
| **File Operations** | Safely read and write code files within a workspace |
| **Traceability** | LangSmith integration for full request chain observability |

## Quick Start

### 1. Install

```bash
git clone https://github.com/DZCD/DevMate.git
cd DevMate
uv sync
```

### 2. Configure

```bash
cp config.toml.example config.toml
```

Edit `config.toml` and add your API keys:

```toml
[model]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_kimi_api_key"
model_name = "kimi-k2.5"

[search]
tavily_api_key = "your_tavily_api_key"
```

### 3. Start Services

**Option 1: Using Docker Compose (Recommended)**

```bash
docker compose up -d --build && docker compose run --rm devmate chat
```

**Option 2: Local Run**

```bash
# Terminal 1: Start MCP Server
uv run devmate serve

# Terminal 2: Start interactive chat
uv run devmate chat
```

### 4. Start Chatting

```bash
# Interactive mode
uv run devmate chat

# Single task
uv run devmate run "Create a FastAPI Hello World service"
```

## Architecture

```
┌─────────────────────────────────────┐
│           CLI (click)               │
│   init / chat / run / serve         │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│         DevMate Agent               │
│  ┌─────────┬─────────┬──────────┐   │
│  │MCP Tools│RAG Tools│Skill Tool│   │
│  └────┬────┴────┬────┴────┬─────┘   │
└───────┼─────────┼─────────┼─────────┘
        ▼         ▼         ▼
   ┌─────────┐ ┌──────┐ ┌────────┐
   │Tavily   │ │Chroma│ │.skills/│
   │(Search) │ │(RAG) │ │(Templates)
   └─────────┘ └──────┘ └────────┘
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `devmate init` | Initialize RAG document index |
| `devmate chat` | Start interactive chat session |
| `devmate run "prompt"` | Execute a single task |
| `devmate serve` | Start MCP Search Server |
| `devmate -v chat` | Verbose logging mode |

## Preview Generated Projects

Generated projects are stored in the `output/` directory by default. Here's how to run one:

```bash
# Navigate to the generated project and start a local server
cd ./output/hiking-website-20260410-leaflet && python3 -m http.server 8888

# Open in browser
# http://localhost:8888
```

## Skills System

Skills are reusable development templates stored in the `.skills/` directory.

**Example skill file** (`.skills/fastapi/SKILL.md`):

```markdown
---
name: "create_fastapi_service"
description: "Create a FastAPI service with standard structure"
trigger_keywords:
  - "fastapi"
  - "api service"
---

## Steps
1. Create project structure
2. Add dependencies
3. Implement main module
```

The agent automatically matches skills based on keywords in user requests.

## RAG Knowledge Base

Place Markdown documents in the `docs/` directory and run `devmate init` to build the index.

The system will automatically:
1. Parse documents by header structure
2. Split into semantic chunks
3. Store in ChromaDB for retrieval

**Configuration example:**

```toml
[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"
chunk_size = 1000
chunk_overlap = 200
```

## Configuration

### Model Configuration

```toml
[model]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_api_key"
model_name = "kimi-k2.5"
temperature = 0.7
max_tokens = 4096
```

Supports any OpenAI-compatible API: Kimi, DeepSeek, OpenAI, Azure, etc.

### Vision Model (Optional)

For image understanding. Defaults to main model config if not specified:

```toml
[vision]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_api_key"
model_name = "kimi-k2.5"
```

### LangSmith Observability

```toml
[langsmith]
enabled = true
langchain_api_key = "your_langsmith_key"
langchain_project = "devmate"
```

## Docker Deployment

```bash
# Build and start all services, then enter interactive chat
docker compose up -d --build && docker compose run --rm devmate chat
```

Then enter your prompt:

```
Please refer to /app/workspace/design-screenshot.png as design reference and generate a frontend website.
```

**Common commands:**

```bash
# Start services (detached)
docker compose up -d

# Run interactive chat
docker compose run --rm devmate chat

# Execute single task
docker compose run --rm devmate run "Create a React project"
```

**Services:**
- `chromadb`: Vector database (port 8000)
- `mcp-server`: MCP service (port 8001)
- `devmate`: Main application container

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format src/ mcp_server/ tests/

# Lint check
uv run ruff check src/ mcp_server/ tests/
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangChain |
| LLM | OpenAI-compatible API |
| MCP Transport | Streamable HTTP |
| Web Search | Tavily API |
| Vector Store | ChromaDB |
| Observability | LangSmith |
| CLI | Click |
| Package Manager | uv |

## License

MIT
