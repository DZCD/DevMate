"""Agent core module for DevMate.

Implements a custom tool loop (no LangGraph/LangChain agent dependency).
Follows the architecture of agent-template-ts/src/agent/createAgent.ts.

Core design:
- Self-built tool loop: messages -> LLM chat -> check tool_calls -> execute
  tools -> append tool_result -> continue loop
- Storage layer for conversation memory (FileStorage with local JSON files)
- LLM abstraction via OpenAI-compatible adapter (DeepSeek)
- Tool registry and executor wrapping existing @tool functions
"""

import asyncio
import logging
from typing import Any

from devmate.config import (
    get_model_config,
    get_rag_config,
    get_skills_config,
    load_config,
)
from devmate.file_tools import create_file_tools
from devmate.llm import LLMToolDef, OpenAICompatibleAdapter  # noqa: E501
from devmate.rag import RAGEngine, create_search_tool
from devmate.skills import SkillsManager
from devmate.storage import (
    ContentBlock,
    Message,
    Storage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    add_message,
    assistant_message,
    create_storage,
    get_beijing_date,
    get_messages,
    tool_result,
    user_message,
)
from devmate.tools import (
    Tool,
    ToolExecutor,
    ToolRegistry,
    langchain_tool_to_tool,
    tools_to_llm_defs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt (aligned with TS template)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an interactive tool that helps users with tasks. Use the instructions below and the tools available to you to assist the user.

<Professional objectivity>
Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation. It is best for the user if Claude honestly applies the same rigorous standards to all ideas and disagrees when necessary, even if it may not be what the user wants to hear. Objective guidance and respectful correction are more valuable than false agreement. Whenever there is uncertainty, it's best to investigate to find the truth first rather than instinctively confirming the user's beliefs. Avoid using over-the-top validation or excessive praise when responding to users such as "You're absolutely right" or similar phrases.
</Professional objectivity>

<Task Management>
You have access to the task tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.
It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
</Task Management>

<Doing tasks>
# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- Use the Task tool to plan the task if required
- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are automatically added by the system, and bear no direct relation to the specific tool results or user messages in which they appear.
</Doing tasks>

<Tool usage policy>
- When doing file search, prefer to use the Task tool in order to reduce context usage.
- When WebFetch returns a message about a redirect to a different host, you should immediately make a new WebFetch request with the redirect URL provided in the response.
- You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel. Maximize use of parallel tool calls where possible to increase efficiency. However, if some tool calls depend on previous calls to inform dependent values, do NOT call these tools in parallel and instead call them sequentially. For instance, if one operation must complete before another starts, run these operations sequentially instead. Never use placeholders or guess missing parameters in tool calls.
- Use specialized tools instead of bash commands when possible, as this provides a better user experience. For file operations, use dedicated tools: Read for reading files instead of cat/head/tail, Edit for editing instead of sed/awk, and Write for creating files instead of cat with heredoc or echo redirection. Reserve bash tools exclusively for actual system commands and terminal operations that require shell execution. NEVER use bash echo or other command-line tools to communicate thoughts, explanations, or instructions to the user. Output all communication directly in your response text instead.
- VERY IMPORTANT: When exploring the codebase to gather context or to answer a question that is not a needle query for a specific file/class/function, it is CRITICAL that you use the Task tool instead of running search commands directly.
- send_message tool is your only way to communicate with the user. You MUST call it at least once per interaction, otherwise the user will not receive your response.
</Tool usage policy>

<Workspace>
Your default working directory is: {workspace_path}
When you need to create files, write documents, generate reports, etc., please place them in this directory (or subdirectories) by default, rather than in the code repository directory.
You can freely read and write files at any path on the system. This workspace is only a suggested default output directory.
</Workspace>

{skills_section}"""


class DevMateAgent:
    """The main DevMate agent with a custom tool loop.

    Follows the architecture of agent-template-ts/src/agent/createAgent.ts.
    """

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
        self._llm: OpenAICompatibleAdapter | None = None
        self._storage: Storage[Any] | None = None
        self._rag_engine: RAGEngine | None = None
        self._skills_manager: SkillsManager | None = None
        self._tool_registry: ToolRegistry | None = None
        self._tool_executor: ToolExecutor | None = None
        self._tools: list[Tool] = []
        self._llm_tool_defs: list[LLMToolDef] = []
        self._system_prompt: str = ""
        self._max_iterations: int = 50
        logger.info("DevMate agent initialized")

    async def initialize(self) -> None:
        """Set up all components: LLM, storage, tools, RAG, skills."""
        logger.info("Initializing DevMate agent components...")

        # 1. Initialize LLM (OpenAI-compatible adapter for DeepSeek)
        model_config = get_model_config(self._config)
        self._llm = OpenAICompatibleAdapter(
            api_key=model_config.get("api_key", ""),
            base_url=model_config.get("base_url", "https://api.deepseek.com"),
            model=model_config.get("model_name", "deepseek-chat"),
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 8192),
        )
        logger.info("LLM initialized: %s", model_config.get("model_name"))

        # 2. Initialize Storage (FileStorage with local JSON files)
        self._storage = create_storage()
        logger.info("Storage initialized")

        # 3. Initialize RAG
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

        # 4. Initialize Skills
        skills_config = get_skills_config(self._config)
        self._skills_manager = SkillsManager(
            skills_dir=skills_config.get("directory", ".skills")
        )
        skills_count = self._skills_manager.load_skills()
        logger.info("Skills loaded: %d", skills_count)

        # 5. Build tool list using ToolRegistry
        self._tool_registry = ToolRegistry()
        file_tools = create_file_tools(self._workspace)
        for lc_tool in file_tools:
            self._tool_registry.register(langchain_tool_to_tool(lc_tool))

        if self._rag_engine:
            search_tool = create_search_tool(self._rag_engine)
            self._tool_registry.register(langchain_tool_to_tool(search_tool))

        skill_tools = (
            self._skills_manager.create_tools() if self._skills_manager else []
        )
        for lc_tool in skill_tools:
            self._tool_registry.register(langchain_tool_to_tool(lc_tool))

        self._tools = self._tool_registry.get_all()
        self._tool_executor = ToolExecutor(self._tool_registry)
        self._llm_tool_defs = tools_to_llm_defs(self._tools)
        logger.info("Tools registered: %d", len(self._tools))

        # 6. Connect to MCP server for additional tools
        await self._connect_mcp()

        # 7. Build system prompt
        self._build_system_prompt()
        logger.info("DevMate agent fully initialized")

    async def _connect_mcp(self) -> None:
        """Connect to MCP server and retrieve additional tools."""
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

            mcp_client = MultiServerMCPClient(connections=connections)
            mcp_tools = await mcp_client.get_tools()

            if mcp_tools:
                for lc_tool in mcp_tools:
                    try:
                        self._tool_registry.register(langchain_tool_to_tool(lc_tool))
                    except ValueError:
                        logger.debug(
                            "MCP tool already registered, skipping: %s", lc_tool.name
                        )
                self._tools = self._tool_registry.get_all()
                self._llm_tool_defs = tools_to_llm_defs(self._tools)
                logger.info(
                    "Connected to MCP server, total tools: %d",
                    len(self._tools),
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

    def _build_system_prompt(self) -> None:
        """Build the full system prompt with skills section."""
        import os

        workspace_path = self._workspace or os.path.join(
            os.path.expanduser("~"), ".duclaw", "workspace"
        )

        skills_section = ""
        if self._skills_manager:
            skill_meta = self._skills_manager.get_skill_meta()
            if skill_meta:
                skills_section = (
                    f"\n<Available Skills>\n{skill_meta}\n</Available Skills>\n"
                )

        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            workspace_path=workspace_path,
            skills_section=skills_section,
        )

    async def run(self, prompt: str, user_id: str = "default") -> str:
        """Run the agent with a given prompt using the custom tool loop.

        This is the core tool loop, following the TS template pattern:
        1. Add user message to storage
        2. While loop: get_messages -> llm.chat -> check tool_calls ->
           execute tools -> add_message(tool_results) -> continue or return

        Args:
            prompt: The user prompt to process.
            user_id: The user identifier for conversation isolation.

        Returns:
            The agent's final text response.
        """
        if self._llm is None:
            await self.initialize()

        logger.info("Running agent with prompt: %s (user=%s)", prompt[:100], user_id)

        try:
            # 1. Get current date for storage key isolation
            date = get_beijing_date()

            # 2. Add user message to storage
            await add_message(self._storage, user_id, user_message(prompt), date)

            # 3. Tool loop
            iterations = 0
            while iterations < self._max_iterations:
                iterations += 1
                logger.debug(
                    "Tool loop iteration %d/%d", iterations, self._max_iterations
                )

                # 3.1 Load message history
                messages = await get_messages(
                    self._storage, user_id, limit=100, date=date
                )

                if not messages:
                    logger.warning("No messages loaded for user %s", user_id)
                    break

                # 3.2 Call LLM
                response = await self._llm.chat(
                    messages=messages,
                    system_prompt=self._system_prompt,
                    tools=self._llm_tool_defs,
                )

                # 3.3 Check for max_tokens truncation
                if response.finish_reason == "length":
                    truncated_tool_uses = [
                        b for b in response.content if isinstance(b, ToolUseBlock)
                    ]
                    if truncated_tool_uses:
                        # Skip truncated tool calls, warn the LLM
                        text_content = (
                            "\n".join(
                                b.text
                                for b in response.content
                                if isinstance(b, TextBlock)
                            )
                            or "(output truncated)"
                        )
                        await add_message(
                            self._storage,
                            user_id,
                            assistant_message(TextBlock(text=text_content)),
                            date,
                        )
                        await add_message(
                            self._storage,
                            user_id,
                            user_message(
                                "<system-warning>Your output was truncated due to max_tokens limit. "
                                "Tool call parameters are incomplete and have been skipped. "
                                "Please reduce output per turn: output text first, then call tools "
                                "separately.</system-warning>"
                            ),
                            date,
                        )
                        continue

                # 3.4 Check for tool calls
                tool_calls = response.tool_calls
                if tool_calls:
                    # Add assistant message (with tool_use blocks) to storage
                    assistant_blocks: list[ContentBlock] = []
                    for block in response.content:
                        assistant_blocks.append(block)
                    await add_message(
                        self._storage,
                        user_id,
                        Message(role="assistant", content=assistant_blocks),
                        date,
                    )

                    # Execute all tools concurrently
                    tool_results: list[ToolResultBlock] = []
                    for tc in tool_calls:
                        try:
                            result = await self._tool_executor.execute(
                                tc.name, tc.arguments
                            )
                            tool_results.append(tool_result(tc.id, result))
                        except Exception as exc:
                            logger.error("Tool %s execution error: %s", tc.name, exc)
                            tool_results.append(
                                tool_result(tc.id, f"Tool [{tc.name}] error: {exc}")
                            )

                    # Add tool results as a single user message
                    if tool_results:
                        await add_message(
                            self._storage,
                            user_id,
                            Message(role="user", content=tool_results),
                            date,
                        )

                    continue  # Continue the loop for LLM to process tool results

                # 3.5 No tool calls - extract text response and return
                text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
                if text_blocks:
                    final_text = "\n".join(b.text for b in text_blocks)
                    # Save the final assistant response to storage
                    await add_message(
                        self._storage,
                        user_id,
                        assistant_message(TextBlock(text=final_text)),
                        date,
                    )
                    logger.info(
                        "Agent response received (%d chars, %d iterations)",
                        len(final_text),
                        iterations,
                    )
                    return final_text

                # No content at all
                logger.warning(
                    "Agent produced no content after %d iterations", iterations
                )
                return "No response generated."

            logger.warning("Agent reached max iterations (%d)", self._max_iterations)
            return f"Reached maximum iterations ({self._max_iterations}). Please try a simpler request."

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc, exc_info=True)
            return f"Error: Agent execution failed - {exc}"

    async def chat_loop(self) -> None:
        """Start an interactive chat loop with conversation memory."""
        await self.initialize()
        user_id = "interactive"
        logger.info("Starting interactive chat session (user=%s)...", user_id)

        while True:
            try:
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

            response = await self.run(user_input, user_id=user_id)
            logger.info("Agent: %s", response)

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Agent cleanup complete")


def create_agent_func(
    config_path: str | None = None,
    workspace: str | None = None,
) -> DevMateAgent:
    """Create and return a DevMate agent instance.

    Args:
        config_path: Path to the config file.
        workspace: Workspace directory path.

    Returns:
        A DevMateAgent instance.
    """
    return DevMateAgent(config_path=config_path, workspace=workspace)
