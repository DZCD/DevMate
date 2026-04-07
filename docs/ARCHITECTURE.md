# DevMate 架构设计文档

## 1. 项目概述

### 1.1 DevMate 是什么

DevMate 是一个基于 AI 的智能开发助手，旨在帮助开发者完成编码、文档撰写、调试和项目管理等任务。它通过整合大语言模型（LLM）、本地知识库检索（RAG）、可复用的技能模板（Skills）以及实时网络搜索能力，为开发者提供一站式的智能辅助体验。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **交互式对话** | 通过 `devmate chat` 启动终端交互式会话，支持多轮对话 |
| **单次任务执行** | 通过 `devmate run "prompt"` 执行单个任务并返回结果 |
| **实时网络搜索** | 通过 MCP Server 集成 Tavily API，获取实时网络信息 |
| **本地知识库（RAG）** | 基于 ChromaDB 的向量检索系统，索引 `docs/` 目录下的 Markdown 文档 |
| **技能系统（Skills）** | 基于 Markdown 文件的插件式知识模板，通过关键词匹配自动触发 |
| **文件操作** | 提供安全的沙箱化文件创建、写入和目录浏览能力 |
| **可观测性** | 集成 LangSmith，支持 Agent 调用链追踪与调试 |

### 1.3 技术栈

| 组件 | 技术 | 版本要求 |
|------|------|----------|
| 语言 | Python | >= 3.13 |
| Agent 框架 | LangChain (create_agent) | >= 1.2.10 |
| LLM | DeepSeek (通过 ChatOpenAI 兼容接口) | - |
| MCP 协议 | Streamable HTTP (无状态模式) | - |
| 网络搜索 | Tavily API | - |
| 向量数据库 | ChromaDB (本地持久化) | - |
| Embedding | OpenAI 兼容接口 (text-embedding-3-small) | - |
| ASGI 框架 | Starlette + Uvicorn | - |
| CLI 框架 | Click | - |
| 依赖管理 | uv + hatchling | - |
| 容器化 | Docker + Docker Compose | - |
| 代码质量 | Ruff (lint + format) | >= 0.6 |
| 测试 | pytest + pytest-asyncio | >= 8.0 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户 (终端)                                │
│                    devmate init / chat / run / serve                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        CLI 入口 (Click)                              │
│                   src/devmate/__main__.py                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
   │   init 命令   │   │ chat/run 命令 │   │   serve 命令      │
   │ (RAG 索引)   │   │ (Agent 对话)  │   │ (启动 MCP Server) │
   └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
          │                  │                    │
          ▼                  ▼                    ▼
   ┌──────────────┐   ┌──────────────────┐  ┌──────────────────┐
   │  RAGEngine   │   │  DevMateAgent    │  │  MCP Server      │
   │  (ChromaDB)  │   │  (LangChain)     │  │  (Starlette)     │
   └──────────────┘   └───────┬──────────┘  └────────┬─────────┘
                              │                      │
              ┌───────────────┼───────────────┐       │
              │               │               │       │
              ▼               ▼               ▼       │
      ┌──────────────┐ ┌────────────┐ ┌──────────┐   │
      │ File Tools   │ │ RAG Tool   │ │  Skills  │   │
      │ (文件操作)    │ │(知识库搜索) │ │ (技能查询)│   │
      └──────────────┘ └─────┬──────┘ └──────────┘   │
                             │                      │
                    ┌────────▼────────┐              │
                    │    ChromaDB     │              │
                    │  (向量数据库)    │              │
                    └─────────────────┘              │
                                                    │
              ┌─────────────────────────────────────┘
              │ (Streamable HTTP :8001/mcp)
              ▼
      ┌──────────────────┐
      │   MCP Server     │
      │  search_web 工具  │
      │  (Tavily API)    │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   Tavily Search  │
      │   (互联网搜索)    │
      └──────────────────┘
```

### 2.2 模块关系

DevMate 采用**松耦合模块化架构**，各模块职责清晰、独立可测：

```
┌─────────────────────────────────────────────────────┐
│                    DevMateAgent                      │
│  (src/devmate/agent.py)                             │
│                                                     │
│  聚合层：将所有工具统一注册到 LangChain Agent          │
├─────────────┬──────────────┬───────────┬────────────┤
│ FileTools   │ RAGEngine    │ SkillsMgr │ MCP Client │
│ file_tools  │ rag.py       │ skills.py │ (外部连接)  │
└──────┬──────┴──────┬───────┴─────┬─────┴─────┬──────┘
       │             │             │           │
       ▼             ▼             ▼           ▼
   文件系统       ChromaDB     .skills/     MCP Server
                               (Markdown)   (HTTP)
```

**模块依赖关系：**

- `__main__.py` (CLI) → `agent.py` (Agent 核心) / `rag.py` (初始化索引) / `config.py` (配置加载)
- `agent.py` → `config.py`, `rag.py`, `skills.py`, `file_tools.py`, `langchain_mcp_adapters`
- `rag.py` → `chromadb`, `langchain_text_splitters`
- `skills.py` → 无外部模块依赖（仅标准库 + langchain_core.tools）
- `file_tools.py` → 无外部模块依赖（仅标准库 + langchain_core.tools）
- `mcp_server/` → `config.py`, `mcp`, `starlette`, `tavily-python`

### 2.3 数据流

以下是用户发起一次交互请求的完整数据流：

```
用户输入 "帮我搜索最新的 FastAPI 文档"
        │
        ▼
[1] CLI 接收命令 (devmate chat / run)
        │
        ▼
[2] DevMateAgent.run(prompt) 被调用
        │
        ▼
[3] LangChain Agent 接收 HumanMessage
        │
        ▼
[4] LLM (DeepSeek) 进行推理，决定调用工具
        │  (基于 SYSTEM_PROMPT 中的决策框架)
        │
        ├─→ [5a] 调用 search_knowledge_base (RAG)
        │         │
        │         ▼
        │    RAGEngine.search(query)
        │         │
        │         ▼
        │    ChromaDB 向量相似度检索
        │         │
        │         ▼
        │    返回相关文档片段
        │
        ├─→ [5b] 调用 search_web (MCP)
        │         │
        │         ▼
        │    MultiServerMCPClient → HTTP POST :8001/mcp
        │         │
        │         ▼
        │    MCP Server (Starlette) → TavilyClient.search()
        │         │
        │         ▼
        │    返回网络搜索结果
        │
        ├─→ [5c] 调用 query_skills (Skills)
        │         │
        │         ▼
        │    SkillsManager.find_matching_skills(query)
        │         │
        │         ▼
        │    关键词匹配，返回匹配的技能内容
        │
        ├─→ [5d] 调用 create_file / write_file / list_directory
        │         │
        │         ▼
        │    在 workspace 范围内执行文件操作
        │
        ▼
[6] LLM 汇总所有工具返回结果，生成最终回复
        │
        ▼
[7] Agent 提取最后一条 AI Message 返回给用户
```

**决策框架**（SYSTEM_PROMPT 中定义）：
1. 优先查询本地知识库（search_knowledge_base）
2. 知识库不足时搜索网络（search_web）
3. 适用时引用技能模板（query_skills）
4. 需要时执行文件操作（create_file / write_file / list_directory）

---

## 3. 模块设计

### 3.1 Agent 模块 (`src/devmate/agent.py`)

#### 概述

Agent 模块是 DevMate 的核心，负责初始化所有组件、聚合工具并驱动 LangChain Agent 完成用户任务。

#### 类：`DevMateAgent`

**初始化流程 (`__init__`)：**
- 加载配置文件 (`load_config`)
- 初始化成员变量：`_llm`、`_agent`、`_rag_engine`、`_skills_manager`、`_tools`、`_mcp_tools`

**完整初始化 (`initialize`)：**

```
initialize() 异步方法执行顺序：
  1. 初始化 LLM ─── ChatOpenAI(base_url, api_key, model_name, temperature, max_tokens)
  2. 初始化 RAG ─── RAGEngine(persist_directory, chunk_size, chunk_overlap, embedding)
                    └── 调用 ingest_documents(docs_dir) 索引文档
                    └── 失败时降级为无 Embedding 模式
  3. 初始化 Skills ─ SkillsManager(skills_dir) → load_skills()
  4. 构建工具列表 ─ create_file_tools() + create_search_tool() + skills.create_tools()
  5. 连接 MCP Server ─ MultiServerMCPClient(streamable_http) → get_tools()
  6. 创建 Agent ─── create_agent(model, tools, system_prompt)
```

**MCP 连接 (`_connect_mcp`)：**
- 使用 `langchain-mcp-adapters` 的 `MultiServerMCPClient`
- 传输协议：`streamable_http`
- 连接地址从配置读取：`http://{host}:{port}{route}`
- 连接失败时降级，仅依赖 RAG 进行搜索
- v0.2+ API：`MultiServerMCPClient` 不是上下文管理器，直接调用 `await client.get_tools()` 获取工具

**Agent 创建 (`_create_agent`)：**
- 使用 `langchain.agents.create_agent`
- 将 SYSTEM_PROMPT 与可用工具列表拼接
- 所有工具（File Tools + RAG Tool + Skills Tool + MCP Tools）统一注册

**任务执行 (`run`)：**
- 接收用户 prompt，封装为 `HumanMessage`
- 调用 `agent.ainvoke({"messages": [HumanMessage(content=prompt)]})`
- 从响应的 messages 列表中提取最后一条 AI 消息作为输出

**交互式会话 (`chat_loop`)：**
- 使用 `asyncio.get_event_loop().run_in_executor(None, input, ...)` 实现异步输入
- 支持 `exit`、`quit`、`q` 退出
- 每次输入调用 `run()` 处理

**资源清理 (`cleanup`)：**
- `MultiServerMCPClient` v0.2+ 每次工具调用创建和销毁会话，无需显式清理

### 3.2 MCP Server 模块 (`mcp_server/`)

#### 概述

MCP Server 是一个独立的 ASGI 服务，基于 MCP（Model Context Protocol）提供网络搜索能力。采用 **Streamable HTTP** 传输协议（无状态模式）。

#### 文件结构

- `mcp_server/__init__.py`：核心实现（与 `server.py` 内容相同）
- `mcp_server/server.py`：启动入口（`python -m mcp_server.server`）

#### 核心函数：`create_mcp_app`

**参数：**
- `tavily_api_key`：Tavily 搜索 API 密钥
- `max_results`：最大搜索结果数（默认 5）
- `route`：MCP 端点路由（默认 `/mcp`）

**实现细节：**

```
create_mcp_app() 构建过程：
  1. 创建 MCP Server 实例 ─── Server("devmate-search")
  2. 注册 list_tools 处理器 ─ 返回 [search_web] 工具定义
  3. 注册 call_tool 处理器  ─ 分发到 _execute_search_web()
  4. 创建 StreamableHTTPSessionManager(stateless=True)
  5. 构建 Starlette 应用：
     - Mount(route, app=handle_streamable_http)  ← MCP 端点
     - Route("/health", ...)                     ← 健康检查端点
  6. 配置 lifespan 管理会话生命周期
```

**search_web 工具定义：**
- 输入 Schema：`{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}`
- 执行逻辑：调用 `TavilyClient.search(query, max_results, include_answer=True)`
- 返回格式：Direct Answer + 编号的结果列表（Title、URL、Content）

**健康检查端点：**
- 路径：`GET /health`
- 响应：`{"status": "ok", "service": "devmate-mcp-server"}`

### 3.3 RAG 模块 (`src/devmate/rag.py`)

#### 概述

RAG（Retrieval-Augmented Generation）模块负责文档摄入、分块、Embedding 和语义检索，基于 ChromaDB 实现本地持久化的向量存储。

#### 类：`RAGEngine`

**初始化参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `persist_directory` | str | `.chroma_db` | ChromaDB 持久化目录 |
| `collection_name` | str | `devmate_docs` | ChromaDB 集合名称 |
| `chunk_size` | int | 1000 | 文档分块大小（字符数） |
| `chunk_overlap` | int | 200 | 分块重叠大小（字符数） |
| `embedding_model_name` | str | `text-embedding-3-small` | Embedding 模型名称 |
| `openai_api_key` | str \| None | None | OpenAI 兼容 API 密钥 |
| `openai_api_base` | str \| None | None | OpenAI 兼容 API 基础 URL |

**文档摄入 (`ingest_documents`)：**

```
文档摄入流程：
  docs_directory/
    └── *.md (递归扫描所有 Markdown 文件)
         │
         ▼
    MarkdownHeaderTextSplitter
    (按 #、##、### 标题分割，保留层级元数据)
         │
         ▼
    RecursiveCharacterTextSplitter
    (按 chunk_size/overlap 进一步分块)
         │
         ▼
    ChromaDB collection.upsert()
    (ID 格式: "{文件名}::{块索引}")
    (元数据: source, chunk_index, Header 1, Header 2, Header 3)
```

**Embedding 策略：**
- 当提供 `openai_api_key` 时：使用 `OpenAIEmbeddingFunction`，支持自定义 `api_base`
- 未提供 API Key 时：ChromaDB 使用默认的 Embedding（此时无法进行语义搜索，仅支持基础检索）

**语义检索 (`search`)：**
- 输入：查询字符串 + 最大返回数
- 输出：`list[Document]`（LangChain Document 对象，包含 `page_content` 和 `metadata`）
- 使用 `collection.query(query_texts=[query], n_results=...)` 进行向量相似度搜索
- 空集合时返回空列表

**工具注册 (`create_search_tool`)：**

将 `RAGEngine.search()` 封装为 LangChain `@tool`：
- 工具名：`search_knowledge_base`
- 参数：`query: str`
- 返回格式：编号的文档列表，包含 Source、Section、Content 信息

### 3.4 Skills 系统 (`src/devmate/skills.py`)

#### 概述

Skills 系统是一个基于 Markdown 文件的轻量级知识模板管理器。每个 Skill 定义了可复用的知识模式、代码模板和最佳实践，通过关键词匹配自动触发。

#### 数据结构：`Skill`

```python
@dataclass
class Skill:
    name: str                    # 技能唯一标识
    description: str             # 技能描述
    trigger_keywords: list[str]  # 触发关键词列表
    content: str                 # Markdown 正文（去除 frontmatter）
    source_file: Path | None     # 来源文件路径
```

#### 文件格式

Skills 存储在 `.skills/` 目录下，每个 `.md` 文件为一个 Skill：

```markdown
---
name: "skill_name"
description: "技能描述"
trigger_keywords:
  - "关键词1"
  - "关键词2"
---

技能正文内容（Markdown 格式）
```

#### 类：`SkillsManager`

**核心方法：**

| 方法 | 说明 |
|------|------|
| `load_skills()` | 递归扫描 `.skills/` 目录，加载所有 `.md` 文件 |
| `get_skill(name)` | 按名称获取单个技能 |
| `find_matching_skills(query)` | 基于触发关键词匹配，按匹配分数降序排列 |
| `save_skill(skill)` | 将技能保存为 Markdown 文件（含 frontmatter） |
| `get_all_skills()` | 返回所有已加载技能 |
| `create_tools()` | 创建 LangChain `query_skills` 工具 |

**Frontmatter 解析：**
- 使用正则 `^---\s*\n(.*?)\n---\s*\n` 匹配
- 支持字符串值（带引号/不带引号）和列表值（`[item1, item2]`）
- 支持的 frontmatter 字段：`name`、`description`、`trigger_keywords`

**关键词匹配算法 (`find_matching_skills`)：**
- 将用户查询转为小写
- 对每个 Skill，统计其 `trigger_keywords` 在查询中出现的次数
- 按匹配分数降序排列返回

**工具注册 (`create_tools`)：**

创建一个统一的 `query_skills` 工具，内部调用 `find_matching_skills()`：
- 工具名：`query_skills`
- 参数：`query: str`
- 无匹配时返回可用技能列表

### 3.5 File Tools 模块 (`src/devmate/file_tools.py`)

#### 概述

File Tools 提供沙箱化的文件系统操作能力，所有路径操作限制在 workspace 范围内。

#### 函数：`create_file_tools(workspace)`

返回三个 LangChain 工具：

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `create_file` | `file_path, content, overwrite=False` | 创建新文件，自动创建父目录。默认不覆盖已有文件 |
| `write_file` | `file_path, content` | 写入已有文件（全量替换），文件必须已存在 |
| `list_directory` | `dir_path="."` | 列出目录内容，区分文件和目录，显示文件大小 |

**安全机制：**
- 所有路径通过 `Path.resolve()` 解析为绝对路径
- 路径校验：`str(target).startswith(str(workspace_resolved))`，禁止访问 workspace 外的文件
- 相对路径自动转换为 workspace 内的绝对路径

### 3.6 Config 模块 (`src/devmate/config.py`)

#### 概述

Config 模块负责加载 `config.toml` 配置文件，并提供配置项提取和环境变量设置。

#### 配置加载 (`load_config`)

**配置文件查找顺序：**
1. 显式指定的 `config_path` 参数
2. 当前工作目录下的 `config.toml`
3. 项目根目录下的 `config.toml`（相对于 `config.py` 向上 4 级）

找不到配置文件时抛出 `FileNotFoundError`。

#### LangSmith 集成 (`_apply_langsmith_env`)

当 `langsmith.enabled = true` 时，自动设置以下环境变量：
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<配置中的密钥>`
- `LANGCHAIN_PROJECT=<配置中的项目名>`

未启用时不设置任何环境变量。

#### 配置提取函数

| 函数 | 提取的配置段 |
|------|-------------|
| `get_model_config(config)` | `[model]` 段 |
| `get_search_config(config)` | `[search]` 段 |
| `get_rag_config(config)` | `[rag]` 段 |
| `get_skills_config(config)` | `[skills]` 段 |
| `get_mcp_server_config(config)` | `[mcp_server]` 段 |

### 3.7 CLI 入口 (`src/devmate/__main__.py`)

#### 概述

CLI 入口基于 Click 框架实现，提供 4 个子命令。

#### 命令列表

| 命令 | 说明 | 实现细节 |
|------|------|----------|
| `devmate init` | 初始化文档索引 | 加载 RAG 配置 → 创建 RAGEngine → 调用 ingest_documents → 输出索引统计 |
| `devmate chat` | 启动交互式会话 | 创建 DevMateAgent → `asyncio.run(agent.chat_loop())` → cleanup |
| `devmate run "prompt"` | 执行单次任务 | 创建 DevMateAgent → `asyncio.run(agent.run(prompt))` → 输出结果 → cleanup |
| `devmate serve` | 启动 MCP Server | 加载配置 → 创建 MCP Starlette 应用 → `uvicorn.run()` |

**全局选项：**
- `-v, --verbose`：启用 DEBUG 级别日志
- `--version`：显示版本号

**子命令公共选项：**
- `-c, --config`：指定配置文件路径
- `-w, --workspace`：指定工作目录

**日志配置：**
- 默认输出到 `stderr`
- 格式：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## 4. 技术选型

### 4.1 LLM：DeepSeek (via ChatOpenAI)

DevMate 使用 DeepSeek 作为默认 LLM，通过 LangChain 的 `ChatOpenAI` 适配器连接。`ChatOpenAI` 提供了 OpenAI 兼容接口，支持自定义 `base_url`，因此可以无缝对接任何兼容 OpenAI API 的模型服务。

**选择原因：**
- DeepSeek 在代码生成和中文理解方面表现优秀
- ChatOpenAI 适配器使 LLM 切换成本极低（仅修改 `base_url` 和 `model_name`）
- 支持 Temperature 和 Max Tokens 等生成参数调优

### 4.2 MCP：Streamable HTTP 传输协议

MCP（Model Context Protocol）是 Anthropic 提出的工具调用标准协议。DevMate 采用 **Streamable HTTP** 传输（而非 SSE），基于 `StreamableHTTPSessionManager` 实现无状态会话。

**选择原因：**
- 无状态模式简化了服务端实现，无需维护会话状态
- HTTP 传输天然支持负载均衡和水平扩展
- Streamable HTTP 支持流式和非流式两种响应模式
- 使用 `langchain-mcp-adapters` 实现与 LangChain 的无缝集成

### 4.3 向量数据库：ChromaDB + OpenAI Embedding

**选择原因：**
- ChromaDB 是轻量级嵌入式向量数据库，无需额外服务，适合本地部署
- 使用 SQLite 作为后端存储，数据持久化到 `.chroma_db/` 目录
- 支持 OpenAI 兼容的 Embedding 接口，可对接 DeepSeek 等服务
- `collection.upsert()` 支持幂等更新，适合增量索引

### 4.4 依赖管理：uv + hatchling

**选择原因：**
- uv 是 Rust 编写的高性能 Python 包管理器，安装和解析速度极快
- hatchling 是轻量级构建后端，适合纯 Python 项目
- `pyproject.toml` 统一管理项目元数据和依赖

### 4.5 容器化：Docker + Docker Compose

**选择原因：**
- Docker 确保环境一致性，消除"在我机器上能跑"问题
- Docker Compose 编排 MCP Server 和 Agent 两个服务，支持健康检查和服务依赖
- 配置文件通过 Volume 挂载，不 baked into 镜像，保证密钥安全

---

## 5. 配置说明

### 5.1 config.toml 各配置项详解

```toml
# ==================== LLM 模型配置 ====================
[model]
base_url = "https://api.deepseek.com"      # LLM API 基础 URL（OpenAI 兼容）
api_key = "sk-xxx"                         # LLM API 密钥
model_name = "deepseek-chat"               # 模型名称
embedding_model_name = "text-embedding-3-small"  # Embedding 模型名称
temperature = 0.7                          # 生成温度（0.0-2.0，越高越随机）
max_tokens = 4096                          # 单次生成的最大 Token 数

# ==================== 网络搜索配置 ====================
[search]
tavily_api_key = "tvly-xxx"               # Tavily 搜索 API 密钥
max_results = 5                            # 搜索结果最大数量

# ==================== LangSmith 可观测性 ====================
[langsmith]
enabled = true                             # 是否启用 LangSmith 追踪
langchain_api_key = "lsv2_xxx"            # LangSmith API 密钥
project_name = "devmate"                   # LangSmith 项目名称

# ==================== MCP Server 配置 ====================
[mcp_server]
host = "0.0.0.0"                          # MCP Server 监听地址
port = 8001                               # MCP Server 监听端口
route = "/mcp"                            # MCP 端点路由路径

# ==================== RAG 知识库配置 ====================
[rag]
docs_directory = "docs"                    # 文档目录路径
chroma_persist_directory = ".chroma_db"    # ChromaDB 持久化目录
chunk_size = 1000                         # 文档分块大小（字符数）
chunk_overlap = 200                       # 分块重叠大小（字符数）

# ==================== Skills 技能配置 ====================
[skills]
directory = ".skills"                      # 技能文件目录

# ==================== 输出配置 ====================
[output]
workspace_dir = "./output"                 # 文件操作的默认工作目录
```

### 5.2 环境变量说明

DevMate 不直接依赖环境变量进行配置，所有配置通过 `config.toml` 管理。但以下环境变量由系统内部自动设置：

| 环境变量 | 设置时机 | 说明 |
|----------|----------|------|
| `LANGCHAIN_TRACING_V2` | `config.py` 初始化时 | LangSmith 追踪开关（`true` / 未设置） |
| `LANGCHAIN_API_KEY` | `config.py` 初始化时 | LangSmith API 密钥 |
| `LANGCHAIN_PROJECT` | `config.py` 初始化时 | LangSmith 项目名称 |
| `PYTHONPATH` | Docker Compose 中设置 | `/app/src:/app`，确保模块可导入 |

---

## 6. 部署方案

### 6.1 本地开发部署

**前置条件：** Python 3.13+、uv 包管理器

```bash
# 1. 克隆项目
git clone https://github.com/DZCD/DevMate.git
cd DevMate

# 2. 安装依赖
uv sync

# 3. 配置
cp config.toml.example config.toml
# 编辑 config.toml 填入 API 密钥

# 4. 初始化知识库索引
uv run devmate init

# 5. 启动 MCP Server（终端 1）
uv run devmate serve

# 6. 启动 Agent 交互式会话（终端 2）
uv run devmate chat

# 或执行单次任务
uv run devmate run "帮我创建一个 FastAPI 服务"
```

### 6.2 Docker 容器化部署

**Docker Compose 部署（推荐）：**

```bash
# 构建并启动所有服务
docker compose up --build
```

Docker Compose 编排两个服务：

| 服务 | 容器名 | 说明 |
|------|--------|------|
| `mcp-server` | `devmate-mcp-server` | MCP 搜索服务器，端口 8001 |
| `devmate` | `devmate-agent` | 交互式 Agent，依赖 mcp-server 健康检查通过 |

**Volume 挂载：**
- `./config.toml:/app/config.toml:ro` — 只读挂载配置文件
- `./docs:/app/docs:ro` — 只读挂载文档目录
- `./.skills:/app/.skills:ro` — 只读挂载技能目录

**健康检查：**
- mcp-server：每 10 秒检查 `http://localhost:8001/health`，超时 5 秒，最多重试 5 次
- devmate：通过 `depends_on: condition: service_healthy` 确保 MCP Server 就绪后再启动

**Dockerfile 说明：**
- 基础镜像：`python:3.13-slim`
- 使用 uv 安装依赖：`COPY --from=ghcr.io/astral-sh/uv:latest`
- 先复制 `pyproject.toml` 利用 Docker 缓存层
- 配置文件不 baked into 镜像，运行时通过 Volume 挂载
- 默认命令：`python -m mcp_server.server`（启动 MCP Server）

**单独使用 Dockerfile：**

```bash
docker build -t devmate .
docker run -v ./config.toml:/app/config.toml:ro -p 8001:8001 devmate
```

### 6.3 MCP Server 独立部署

MCP Server 可以独立于 Agent 部署，供任何 MCP 兼容客户端使用：

```bash
# 方式一：通过 CLI 启动
uv run devmate serve --host 0.0.0.0 --port 8001

# 方式二：直接通过 uvicorn
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8001

# 方式三：Docker
docker compose up mcp-server
```

**MCP 端点：** `POST http://<host>:8001/mcp`
**健康检查：** `GET http://<host>:8001/health`

---

## 7. 扩展性设计

### 7.1 如何添加新的 MCP 工具

**方式一：在现有 MCP Server 中添加工具**

1. 在 `mcp_server/__init__.py` 中定义新工具：

```python
def _create_new_tool(...) -> types.Tool:
    return types.Tool(
        name="new_tool",
        description="新工具的描述",
        inputSchema={
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "参数说明"},
            },
            "required": ["param"],
        },
    )

async def _execute_new_tool(param: str) -> list[types.TextContent]:
    # 实现工具逻辑
    return [types.TextContent(type="text", text="结果")]
```

2. 在 `create_mcp_app` 中注册：

```python
new_tool = _create_new_tool(...)

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [search_tool, new_tool]  # 添加新工具

@server.call_tool()
async def call_tool(name, arguments):
    if name == "new_tool":
        return await _execute_new_tool(arguments["param"])
    # ... 其他工具处理
```

**方式二：连接外部 MCP Server**

在 `agent.py` 的 `_connect_mcp` 方法中添加新的连接配置：

```python
connections = {
    "search": {
        "transport": "streamable_http",
        "url": "http://localhost:8001/mcp",
    },
    "external_service": {
        "transport": "streamable_http",
        "url": "http://external-service:8002/mcp",
    },
}
```

Agent 会自动从所有连接的 MCP Server 获取工具并注册。

### 7.2 如何添加新的 Skill

1. 在 `.skills/` 目录下创建新的 Markdown 文件，例如 `.skills/create_api_service.md`：

```markdown
---
name: "create_api_service"
description: "创建标准的 RESTful API 服务"
trigger_keywords:
  - "api service"
  - "rest api"
  - "create api"
---

# 创建 API 服务模板

## 步骤
1. 创建项目结构
2. 实现数据模型
3. 编写路由处理
...
```

2. 重新启动 Agent，Skills 系统会自动加载新文件（无需修改代码）

3. 当用户输入包含触发关键词的查询时，Agent 会自动匹配并引用该 Skill

**也可以通过代码动态添加：**

```python
from devmate.skills import Skill, SkillsManager

manager = SkillsManager(".skills")
manager.save_skill(Skill(
    name="my_skill",
    description="自定义技能",
    trigger_keywords=["关键词"],
    content="技能内容..."
))
```

### 7.3 如何接入其他 LLM

DevMate 通过 `ChatOpenAI` 适配器连接 LLM，任何兼容 OpenAI API 的模型服务都可以直接接入。

**步骤：**

1. 修改 `config.toml` 中的 `[model]` 配置：

```toml
[model]
# 示例：接入 OpenAI 官方
base_url = "https://api.openai.com/v1"
api_key = "sk-xxx"
model_name = "gpt-4o"

# 示例：接入智谱 GLM
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key = "xxx"
model_name = "glm-4"

# 示例：接入 Ollama 本地模型
base_url = "http://localhost:11434/v1"
api_key = "ollama"
model_name = "qwen2.5-coder"
```

2. 如果需要调整 Embedding 模型，修改 `embedding_model_name`：

```toml
[model]
embedding_model_name = "text-embedding-3-small"  # 或其他兼容的模型名
```

3. 无需修改任何代码，Agent 会在下次启动时使用新配置

**注意事项：**
- 不同模型的工具调用能力有差异，建议使用支持 Function Calling 的模型
- `temperature` 和 `max_tokens` 可能需要根据模型特性调整
- Embedding 模型需要与向量数据库兼容

---

## 附录：项目目录结构

```
DevMate/
├── pyproject.toml              # 项目元数据、依赖、构建配置
├── config.toml                 # 运行时配置（包含 API 密钥，不提交 Git）
├── config.toml.example         # 配置模板（提交 Git）
├── Dockerfile                  # Docker 镜像定义
├── docker-compose.yml          # Docker Compose 服务编排
├── .python-version             # Python 版本锁定
├── .gitignore                  # Git 忽略规则
│
├── src/devmate/                # 核心 Python 包
│   ├── __init__.py             # 包初始化，定义 __version__ = "0.1.0"
│   ├── __main__.py             # CLI 入口 (Click)
│   ├── config.py               # 配置加载器 (TOML)
│   ├── agent.py                # Agent 核心 (LangChain)
│   ├── rag.py                  # RAG 引擎 (ChromaDB)
│   ├── skills.py               # 技能系统 (Markdown)
│   └── file_tools.py           # 文件操作工具
│
├── mcp_server/                 # MCP Server 包
│   ├── __init__.py             # MCP Server 实现 (Streamable HTTP)
│   └── server.py               # Server 启动入口
│
├── docs/                       # RAG 知识库文档
│   ├── ARCHITECTURE.md         # 本文档
│   ├── internal_fastapi_guidelines.md
│   └── project_template_guide.md
│
├── .skills/                    # 技能文件目录
│   ├── .gitkeep
│   └── example_skill.md        # 示例技能
│
├── tests/                      # 测试套件
│   ├── __init__.py
│   ├── conftest.py             # 测试公共 fixtures
│   ├── test_agent.py           # Agent 测试
│   ├── test_config.py          # 配置加载测试
│   ├── test_file_tools.py      # 文件工具测试
│   ├── test_integration.py     # 集成测试
│   ├── test_mcp_server.py      # MCP Server 测试
│   ├── test_rag.py             # RAG 引擎测试
│   ├── test_skills.py          # 技能系统测试
│   └── test_e2e.py             # 端到端测试
│
├── scripts/
│   └── lint.sh                 # 代码检查脚本
│
├── .chroma_db/                 # ChromaDB 持久化数据（自动生成）
├── .pytest_cache/              # pytest 缓存
└── .ruff_cache/                # ruff 缓存
```
