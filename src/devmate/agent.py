"""Agent core module for DevMate.

Creates an agent with MCP tools, RAG retrieval, file operations,
and skills integration. Uses LangChain with DeepSeek LLM via OpenAI-compatible API.
"""

import asyncio
import logging

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from devmate.config import (
    get_model_config,
    get_rag_config,
    get_skills_config,
    load_config,
)
from devmate.file_tools import create_file_tools
from devmate.rag import RAGEngine, create_search_tool
from devmate.skills import SkillsManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DevMate, an AI-powered development assistant. You help
developers with coding tasks, documentation, debugging, and project management.

## Your Capabilities

1. **Web Search** (search_web): Search the internet for current information,
   documentation, and solutions. Use this when you need up-to-date information
   or when the user asks about recent events, new libraries, or APIs.

2. **Knowledge Base** (search_knowledge_base): Search the local knowledge base
   for internal documentation, coding guidelines, and project-specific information.
   Use this to find internal standards, architecture decisions, and best practices.

3. **Skills** (query_skills): Access reusable knowledge patterns and code templates.
   Use this when a task matches a known pattern or when you want to reference
   established practices.

4. **File Operations**:
   - create_file: Create new files with initial content
   - write_file: Update existing files with new content
   - list_directory: Browse directory contents

## Decision Framework

When a user asks a question or gives a task:
1. First, check the knowledge base for relevant internal documentation.
2. If the knowledge base doesn't have sufficient information, search the web.
3. Use skills to find reusable patterns when applicable.
4. Use file tools to create or modify code as needed.

## Response Guidelines

- Be concise and actionable in your responses.
- Always provide working code when asked to generate code.
- Explain your reasoning when making decisions.
- If you're unsure about something, say so rather than guessing.
- Follow the coding standards found in the knowledge base.
- When creating files, use proper project structure.

## Important Notes

- Always use tools when relevant rather than answering from memory.
- When searching, use specific and targeted queries.
- When writing code, follow PEP 8 and project conventions.
- Never use print() statements - use logging instead."""


class DevMateAgent:
    """The main DevMate agent."""

    def __init__(
        self,
        config_path: str | None = None,
        workspace: str | None = None,
    ) -> None:
        """Initialize the DevMate agent.

        Args:
            config_path: Path to the config file.
            workspace: The workspace directory.
        """
        self._config = load_config(config_path)
        self._workspace = workspace
        self._llm = None
        self._agent = None
        self._rag_engine: RAGEngine | None = None
        self._skills_manager: SkillsManager | None = None
        self._tools: list[BaseTool] = []
        self._mcp_tools: list[BaseTool] = []
        logger.info("DevMate agent initialized")

    async def initialize(self) -> None:
        """Set up all components: LLM, tools, RAG, skills, MCP."""
        logger.info("Initializing DevMate agent components...")

        # Initialize LLM
        model_config = get_model_config(self._config)
        self._llm = ChatOpenAI(
            base_url=model_config.get("base_url"),
            api_key=model_config.get("api_key"),
            model=model_config.get("model_name", "deepseek-chat"),
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 4096),
        )
        logger.info("LLM initialized: %s", model_config.get("model_name"))

        # Initialize RAG (skip embedding API if provider doesn't support it)
        rag_config = get_rag_config(self._config)
        embedding_api_key = model_config.get("api_key")
        embedding_api_base = model_config.get("base_url")
        docs_dir = rag_config.get("docs_directory", "docs")
        persist_dir = rag_config.get("chroma_persist_directory", ".chroma_db")

        try:
            self._rag_engine = RAGEngine(
                persist_directory=persist_dir,
                chunk_size=rag_config.get("chunk_size", 1000),
                chunk_overlap=rag_config.get("chunk_overlap", 200),
                embedding_model_name=model_config.get(
                    "embedding_model_name", "text-embedding-3-small"
                ),
                openai_api_key=embedding_api_key,
                openai_api_base=embedding_api_base,
            )
            count = self._rag_engine.ingest_documents(docs_dir)
            logger.info("RAG engine ready (%d chunks indexed)", count)
        except Exception as exc:
            logger.warning(
                "RAG init failed (no embedding API?), continuing without RAG: %s",
                exc,
            )
            self._rag_engine = RAGEngine(
                persist_directory=persist_dir,
                chunk_size=rag_config.get("chunk_size", 1000),
                chunk_overlap=rag_config.get("chunk_overlap", 200),
            )

        # Initialize Skills
        skills_config = get_skills_config(self._config)
        self._skills_manager = SkillsManager(
            skills_dir=skills_config.get("directory", ".skills")
        )
        skills_count = self._skills_manager.load_skills()
        logger.info("Skills loaded: %d", skills_count)

        # Build tool list
        self._tools = []
        self._tools.extend(create_file_tools(self._workspace))
        self._tools.append(create_search_tool(self._rag_engine))
        self._tools.extend(self._skills_manager.create_tools())

        # Connect to MCP server for search tools
        await self._connect_mcp()

        # Create the agent
        self._create_agent()
        logger.info("DevMate agent fully initialized")

    async def _connect_mcp(self) -> None:
        """Connect to MCP server and retrieve tools.

        Uses the langchain-mcp-adapters v0.2+ API where MultiServerMCPClient
        is NOT a context manager. Instead, call ``await client.get_tools()``
        directly — it creates a fresh session per invocation internally.
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            mcp_config = self._config.get("mcp_server", {})
            host = mcp_config.get("host", "localhost")
            port = mcp_config.get("port", 8001)
            route = mcp_config.get("route", "/mcp")

            connections = {
                "search": {
                    "transport": "streamable_http",
                    "url": f"http://{host}:{port}{route}",
                }
            }

            logger.info(
                "Connecting to MCP server at http://%s:%s%s",
                host,
                port,
                route,
            )

            self._mcp_client = MultiServerMCPClient(connections=connections)
            mcp_tools = await self._mcp_client.get_tools()

            if mcp_tools:
                self._tools.extend(mcp_tools)
                self._mcp_tools = mcp_tools
                logger.info(
                    "Connected to MCP server, got %d tools",
                    len(mcp_tools),
                )
            else:
                logger.warning(
                    "MCP server returned no tools. Search will rely on RAG only."
                )

        except Exception as exc:
            logger.warning(
                "Failed to connect to MCP server: %s. Continuing without MCP tools.",
                exc,
            )
            self._mcp_client = None

    def _create_agent(self) -> None:
        """Create the agent with all tools using langchain create_agent."""
        tool_names = ", ".join(t.name for t in self._tools)
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools
        )

        system_prompt = (
            SYSTEM_PROMPT
            + "\n\n## Available Tools\n\n"
            + tool_names
            + "\n\n"
            + tool_descriptions
        )

        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )
        logger.info("Agent created with %d tools", len(self._tools))

    async def run(self, prompt: str) -> str:
        """Run the agent with a given prompt.

        Args:
            prompt: The user prompt to process.

        Returns:
            The agent's response.
        """
        if self._agent is None:
            await self.initialize()

        logger.info("Running agent with prompt: %s", prompt[:100])

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )
            # Extract the last AI message content from the response
            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.type == "ai":
                    output = msg.content
                    break
            if not output:
                output = "No output generated."
            logger.info("Agent response received (%d chars)", len(output))
            return output
        except Exception as exc:
            logger.error("Agent execution failed: %s", exc, exc_info=True)
            return f"Error: Agent execution failed - {exc}"

    async def chat_loop(self) -> None:
        """Start an interactive chat loop."""

        await self.initialize()
        logger.info("Starting interactive chat session...")

        while True:
            try:
                # Use asyncio-compatible input
                loop = asyncio.get_event_loop()
                user_input = await loop.run_in_executor(None, input, "\nYou: ")
            except (EOFError, KeyboardInterrupt):
                logger.info("Chat session ended")
                break

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("exit", "quit", "q"):
                logger.info("Chat session ended by user")
                break

            response = await self.run(user_input)
            # Use logging instead of print for the response
            logger.info("Agent: %s", response)

    async def cleanup(self) -> None:
        """Clean up resources.

        MultiServerMCPClient (v0.2+) is not a context manager and has no
        explicit close method — sessions are created and torn down per tool
        call, so no cleanup is needed.
        """
        logger.info("Agent cleanup complete")


def create_agent_func(
    config_path: str | None = None,
    workspace: str | None = None,
) -> DevMateAgent:
    """Create and return a DevMate agent instance.

    Args:
        config_path: Path to the config file.
        workspace: The workspace directory.

    Returns:
        A DevMateAgent instance.
    """
    return DevMateAgent(config_path=config_path, workspace=workspace)
