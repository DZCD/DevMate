"""Patch for MCP tools to fix parameter passing issues."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def wrap_mcp_tool(mcp_tool):
    """Wrap MCP tool to fix argument passing issues."""
    original_arun = mcp_tool.arun

    async def patched_arun(*args, **kwargs):
        # Fix: langchain-mcp-adapters sometimes passes args instead of kwargs
        if not kwargs and args and isinstance(args[0], dict):
            kwargs = args[0]
            args = ()
        elif 'query' not in kwargs and args and isinstance(args[0], str):
            kwargs = {'query': args[0]}
            args = ()

        logger.debug("Calling %s with kwargs=%s", mcp_tool.name, kwargs)
        return await original_arun(*args, **kwargs)

    mcp_tool.arun = patched_arun
    return mcp_tool
