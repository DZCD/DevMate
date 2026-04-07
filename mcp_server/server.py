"""MCP Server entry point.

Start the MCP server with:
    uvicorn mcp_server.server:app --host 0.0.0.0 --port 8001
"""

import logging
import sys

import uvicorn

from devmate.config import get_mcp_server_config, get_search_config, load_config
from mcp_server import create_mcp_app

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting DevMate MCP Server...")

    try:
        config = load_config()
        search_config = get_search_config(config)
        mcp_config = get_mcp_server_config(config)

        tavily_api_key = search_config.get("tavily_api_key", "")
        max_results = search_config.get("max_results", 5)
        host = mcp_config.get("host", "0.0.0.0")
        port = mcp_config.get("port", 8001)
        route = mcp_config.get("route", "/mcp")

        if not tavily_api_key or tavily_api_key.startswith("YOUR_"):
            logger.warning(
                "Tavily API key not configured. "
                "Set 'tavily_api_key' in config.toml [search] section."
            )

        app = create_mcp_app(
            tavily_api_key=tavily_api_key,
            max_results=max_results,
            route=route,
        )

        logger.info("MCP Server listening on http://%s:%s%s", host, port, route)
        uvicorn.run(app, host=host, port=port)

    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to start MCP Server: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
