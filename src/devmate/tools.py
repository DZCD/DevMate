"""Tool registry and executor for DevMate.

Provides ToolRegistry (Map of tools) and ToolExecutor (validates and runs tools).

Follows the architecture of agent-template-ts/src/tools/.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from devmate.llm import LLMToolDef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """A tool definition with execution function.

    Mirrors agent-template-ts/src/tools/Tool.ts.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Tool registry (maps tool names to Tool instances).

    Mirrors agent-template-ts/src/tools/ToolRegistry.ts.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Validates inputs and executes tools.

    Mirrors agent-template-ts/src/tools/ToolExecutor.ts.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Execute a tool by name with the given arguments.

        Args:
            name: The tool name.
            arguments: Tool arguments (dict).

        Returns:
            The tool execution result as a string.
        """
        tool = self._registry.get(name)
        if tool is None:
            return f"Tool not found: {name}"

        args = arguments or {}

        # Validate required fields from JSON Schema
        required = tool.parameters.get("required", [])
        for key in required:
            if key not in args or args[key] is None:
                return f"Tool {name} missing required parameter: {key}"

        try:
            return await tool.execute(**args)
        except Exception as exc:
            logger.error("Tool %s execution failed: %s", name, exc)
            return f"Tool [{name}] execution failed: {exc}"


# ---------------------------------------------------------------------------
# Helper: convert LangChain @tool to our Tool
# ---------------------------------------------------------------------------


def langchain_tool_to_tool(lc_tool: Any) -> Tool:
    """Convert a LangChain @tool-decorated function to our Tool type.

    Args:
        lc_tool: A LangChain tool instance (BaseTool).

    Returns:
        A Tool instance wrapping the LangChain tool.
    """

    async def _execute(**kwargs: Any) -> str:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Tool {lc_tool.name} called with kwargs: {kwargs}")
        # Fix: Try multiple formats for MCP tools compatibility
        if hasattr(lc_tool, "ainvoke"):
            try:
                result = await lc_tool.ainvoke(kwargs)
            except Exception as e:
                logger.error(f"Tool {lc_tool.name} failed: {e}")
                raise
        else:
            result = lc_tool.invoke(kwargs)
        return str(result) if result is not None else ""

    # Extract JSON Schema from the LangChain tool
    try:
        schema = lc_tool.args_schema.model_json_schema() if lc_tool.args_schema else {}
    except Exception:
        schema = {}

    # Build parameters in JSON Schema format
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Remove $defs and other non-standard top-level keys
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    # Handle description - try both attribute and function docstring
    description = getattr(lc_tool, "description", "") or ""

    return Tool(
        name=lc_tool.name,
        description=description,
        parameters=parameters,
        execute=_execute,
    )


def tools_to_llm_defs(tools: list[Tool]) -> list[LLMToolDef]:
    """Convert a list of Tool instances to LLMToolDef for the LLM API."""
    return [
        LLMToolDef(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )
        for tool in tools
    ]
