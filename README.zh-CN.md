# DevMate

[English version](./README.md)

一个集成了 MCP、RAG 和 Skills 的 AI 编程助手。

## 功能特性

- **联网搜索**：通过 MCP（Model Context Protocol）和 Streamable HTTP 传输接入 Tavily，实现实时 Web Search
- **知识库检索（RAG）**：使用 ChromaDB 对本地文档进行索引和语义检索
- **技能系统（Skills）**：将可复用的知识模式和代码模板存放为 Markdown 技能文件
- **文件操作**：在受控工作区内创建、写入和浏览文件
- **LangSmith 集成**：支持 Agent Trace、调用链观测与调试
- **模块化架构**：MCP Server、RAG 引擎、Agent 核心和 CLI 清晰解耦

## 架构概览

```text
CLI (click)
└── devmate init / chat / run / serve
    └── DevMate Agent
        ├── MCP Tools
        ├── RAG Tool
        ├── Skill Tools
        └── File Tools
            ├── MCP Server (Tavily)
            ├── ChromaDB
            ├── .skills/
            └── Workspace File System
```

## 环境要求

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) 包管理器
- Tavily API Key
- 可用的 OpenAI-compatible / Anthropic-compatible LLM API

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/DZCD/DevMate.git
cd DevMate
uv sync
```

### 2. 配置 `config.toml`

```bash
cp config.toml.example config.toml
# 编辑 config.toml，填入你的 API Key
```

最小示例：

```toml
[model]
base_url = "https://api.deepseek.com"
api_key = "your_api_key_here"
model_name = "deepseek-chat"

[search]
tavily_api_key = "your_tavily_api_key_here"
```

### 3. 初始化知识库

```bash
uv run devmate init
```

### 4. 启动 MCP Server（新开一个终端）

```bash
uv run devmate serve
```

### 5. 启动交互式对话

```bash
uv run devmate chat
```

或者执行一次性任务：

```bash
uv run devmate run "Build a FastAPI service for managing hiking trails"
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `devmate init` | 初始化 RAG 文档索引 |
| `devmate chat` | 启动交互式会话 |
| `devmate run "prompt"` | 执行单次任务 |
| `devmate serve` | 启动 MCP Search Server |
| `devmate --version` | 查看版本号 |
| `devmate -v chat` | 以详细日志模式启动 chat |

## Skills 系统

技能文件存放在 `.skills/` 目录下，使用带 YAML frontmatter 的 `SKILL.md` / Markdown 文件来定义，例如：

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

Agent 会根据用户请求中的关键词，自动匹配合适的技能。

## RAG 知识库

将 Markdown 文档放入 `docs/` 目录后，系统会：

1. 使用 `MarkdownHeaderTextSplitter` 按标题结构解析
2. 使用 `RecursiveCharacterTextSplitter` 按配置切分文本块
3. 将内容写入 ChromaDB，供语义检索使用

示例配置：

```toml
[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"
chunk_size = 1000
chunk_overlap = 200
```

## MCP Server

MCP Server 使用 **Streamable HTTP**（无状态模式）传输，核心组件包括：

- `mcp.server.lowlevel.Server`
- `StreamableHTTPSessionManager`
- `Starlette`
- 默认端点：`http://localhost:8001/mcp`

## LangSmith 集成

在 `config.toml` 中配置 LangSmith：

```toml
[langsmith]
enabled = true
langchain_api_key = "your_langsmith_api_key"
langchain_project = "devmate"
```

### 已验证的 Trace 示例

下面这条是一次真实任务执行成功后记录到 LangSmith 的 trace：

- Trace：<https://smith.langchain.com/o/f84fbc14-50a8-44fe-9c85-716ce58215f6/projects/p/3a5251db-8113-439f-9210-4dc44f80828c/r/019d6d56-8abd-72e2-b9de-b294da3b79b5?trace_id=019d6d56-8abd-72e2-b9de-b294da3b79b5&start_time=2026-04-08T13:44:41.406057>
- 验证任务：`Create a minimal FastAPI hello-world service with one /health endpoint.`

## Docker

### 使用 Docker Compose

```bash
docker compose up --build
```

这会启动：

- **chromadb**：Chroma 向量数据库（端口 8000）
- **mcp-server**：MCP Search Server（端口 8001）
- **devmate**：运行 `python -m devmate chat` 的交互式 agent 容器

**交互说明：**

- `devmate` 服务启用了 `stdin_open: true` 和 `tty: true`，设计目标就是交互式 chat。
- `docker compose up --build` 更适合用来启动整套服务并查看日志。
- 如果你想更稳定、直接地进入交互聊天，推荐使用：

```bash
# 先启动整套服务
docker compose up --build

# 在另一个终端进入 devmate 容器，启动交互式 chat
docker compose exec devmate python -m devmate chat

# 或者直接启动一次性 devmate 交互会话
docker compose run --rm devmate chat
```

这样可以把“服务启动”和“实际交互入口”区分得更清楚，也更方便评审验证。

### 直接使用 Dockerfile

```bash
docker build -t devmate .
docker run -v ./config.toml:/app/config.toml:ro -p 8001:8001 devmate
```

## 测试

```bash
# 运行全部测试
uv run pytest tests/ -v

# 运行某个测试模块
uv run pytest tests/test_config.py -v

# 输出更详细日志
uv run pytest tests/ -v -s
```

## 代码质量

```bash
# 格式化
uv run ruff format src/ mcp_server/ tests/

# Lint 检查
uv run ruff check src/ mcp_server/ tests/

# 检查是否误留 print()
grep -rn "print(" src/ mcp_server/ tests/
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent Framework | LangChain |
| LLM | OpenAI-compatible / Anthropic-compatible API |
| MCP Transport | Streamable HTTP |
| Web Search | Tavily API |
| Vector Store | ChromaDB |
| Text Splitting | LangChain Text Splitters |
| Observability | LangSmith |
| CLI | Click |
| ASGI Server | Uvicorn + Starlette |
| Package Manager | uv |
| Linting | Ruff |

## 项目结构

```text
DevMate/
├── pyproject.toml
├── config.toml
├── config.toml.example
├── Dockerfile
├── docker-compose.yml
├── src/devmate/
├── mcp_server/
├── docs/
├── .skills/
└── tests/
```

## License

MIT
