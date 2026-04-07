"""MCP Server implementation using Streamable HTTP transport.

Provides a search_web tool powered by Tavily Search API.
Uses mcp.server.lowlevel.Server with StreamableHTTPSessionManager.
"""

import logging
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount, Route

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp_server = Server("devmate-search")


def _create_search_web_tool(tavily_api_key: str, max_results: int = 5) -> types.Tool:
    """Create the search_web tool definition."""
    return types.Tool(
        name="search_web",
        description=(
            "Search the web for current information using Tavily. "
            "Returns a list of relevant results with titles, URLs, and content "
            "snippets. Use this tool when you need up-to-date information or "
            "when the user asks about recent events, technologies, or documentation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
            },
            "required": ["query"],
        },
    )


async def _execute_search_web(
    query: str, tavily_api_key: str, max_results: int = 5
) -> list[types.TextContent]:
    """Execute a web search using Tavily API."""
    from tavily import TavilyClient

    logger.info("Executing web search: %s", query)
    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )

        # Format results
        results_parts: list[str] = []

        if response.get("answer"):
            results_parts.append(f"Direct Answer: {response['answer']}")

        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            results_parts.append(
                f"\n--- Result {i} ---\nTitle: {title}\nURL: {url}\nContent: {content}"
            )

        if not results_parts:
            return [types.TextContent(type="text", text="No results found.")]

        return [types.TextContent(type="text", text="\n".join(results_parts))]

    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        return [types.TextContent(type="text", text=f"Search error: {exc!s}")]


def create_mcp_app(
    tavily_api_key: str,
    max_results: int = 5,
    route: str = "/mcp",
) -> Starlette:
    """Create the Starlette application with MCP server mounted.

    Args:
        tavily_api_key: Tavily API key for web search.
        max_results: Maximum number of search results.
        route: Route path for MCP endpoint.

    Returns:
        A Starlette application instance.
    """
    logger.info("Creating MCP server app (route=%s)", route)

    search_tool = _create_search_web_tool(tavily_api_key, max_results)

    @mcp_server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available tools."""
        return [search_tool]

    @mcp_server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool calls."""
        arguments = arguments or {}
        if name == "search_web":
            query = arguments.get("query", "")
            if not query:
                return [types.TextContent(type="text", text="Query is required.")]
            return await _execute_search_web(query, tavily_api_key, max_results)
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    # Create session manager with stateless mode
    session_manager = StreamableHTTPSessionManager(app=mcp_server, stateless=True)

    # Build Starlette app
    app = Starlette(
        routes=[
            Mount(
                route,
                app=session_manager.handle_streamable_http,
            ),
            Route("/health", lambda _: _health_response()),
        ],
    )

    return app


def _health_response() -> Any:
    """Return a health check response."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "service": "devmate-mcp-server"})
