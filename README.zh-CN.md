# DevMate

[English version](./README.md)

DevMate 是一个面向开发者的 AI 编程助手，集成 MCP、RAG 和 Skills 系统，让 AI 真正理解你的项目上下文。

## 核心能力

| 能力 | 说明 |
|------|------|
| **联网搜索** | 通过 MCP 接入 Tavily，实时获取网络信息 |
| **知识库检索** | 基于 ChromaDB 的 RAG 系统，让 AI 读懂你的文档 |
| **技能系统** | 可复用的 Prompt 模板，标准化常见开发任务 |
| **文件操作** | 在工作区内安全地读写代码文件 |
| **调用链观测** | LangSmith 集成，可追踪每次请求的完整链路 |

## 快速开始

### 1. 安装

```bash
git clone https://github.com/DZCD/DevMate.git
cd DevMate
uv sync
```

### 2. 配置

```bash
cp config.toml.example config.toml
```

编辑 `config.toml`，填入 API Key：

```toml
[model]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_kimi_api_key"
model_name = "kimi-k2.5"

[search]
tavily_api_key = "your_tavily_api_key"
```

### 3. 启动服务

**方式一：使用 Docker Compose（推荐）**

```bash
docker compose up -d --build
docker compose run --rm devmate chat
```

**方式二：本地运行**

```bash
# 终端 1：启动 MCP Server
uv run devmate serve

# 终端 2：启动交互式对话
uv run devmate chat
```

### 4. 开始对话

```bash
# 交互模式
uv run devmate chat

# 单次任务
uv run devmate run "创建一个 FastAPI Hello World 服务"
```

## 架构

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
   │(Search) │ │(RAG) │ │(模板)  │
   └─────────┘ └──────┘ └────────┘
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `devmate init` | 初始化 RAG 文档索引 |
| `devmate chat` | 启动交互式会话 |
| `devmate run "prompt"` | 执行单次任务 |
| `devmate serve` | 启动 MCP Search Server |
| `devmate -v chat` | 详细日志模式 |

## 预览生成的项目

DevMate 生成的项目默认存放在 `output/` 目录。以下是一个运行示例：

```bash
# 进入生成的项目目录并启动本地服务器
cd ./output/hiking-website-20260410-leaflet && python3 -m http.server 8888

# 浏览器访问
# http://localhost:8888
```

## Skills 系统

Skills 是可复用的开发模板，存放在 `.skills/` 目录。

**示例技能文件**（`.skills/fastapi/SKILL.md`）：

```markdown
---
name: "create_fastapi_service"
description: "创建标准结构的 FastAPI 服务"
trigger_keywords:
  - "fastapi"
  - "api service"
---

## 步骤
1. 创建项目结构
2. 添加依赖
3. 实现主模块
```

Agent 会自动匹配用户请求中的关键词，加载对应的技能。

## RAG 知识库

将 Markdown 文档放入 `docs/` 目录，运行 `devmate init` 即可建立索引。

系统会自动：
1. 按标题结构解析文档
2. 切分为语义块
3. 存入 ChromaDB 供检索

**配置示例**：

```toml
[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"
chunk_size = 1000
chunk_overlap = 200
```

## 配置详解

### 模型配置

```toml
[model]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_api_key"
model_name = "kimi-k2.5"
temperature = 0.7
max_tokens = 4096
```

支持任何 OpenAI-compatible API：Kimi、DeepSeek、OpenAI、Azure 等。

### 视觉模型（可选）

用于图像理解，默认复用主模型配置：

```toml
[vision]
base_url = "https://api.moonshot.cn/v1"
api_key = "your_api_key"
model_name = "kimi-k2.5"
```

### LangSmith 观测

```toml
[langsmith]
enabled = true
langchain_api_key = "your_langsmith_key"
langchain_project = "devmate"
```

## Docker 部署

```bash
# 构建并启动所有服务，然后进入交互式对话
docker compose up -d --build && docker compose run --rm devmate chat
```

然后输入指令：
```
请参考 /app/workspace/徒步网站截图.png 作为设计参考，生成前端网站。
```

**常用命令：**

```bash
# 启动服务（后台运行）
docker compose up -d

# 运行交互式对话
docker compose run --rm devmate chat

# 执行单次任务
docker compose run --rm devmate run "创建一个 React 项目"
```

**服务说明：**
- `chromadb`：向量数据库（端口 8000）
- `mcp-server`：MCP 服务（端口 8001）
- `devmate`：主应用容器

## 开发

```bash
# 运行测试
uv run pytest tests/ -v

# 代码格式化
uv run ruff format src/ mcp_server/ tests/

# Lint 检查
uv run ruff check src/ mcp_server/ tests/
```

## 技术栈

| 组件 | 技术 |
|------|------|
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
