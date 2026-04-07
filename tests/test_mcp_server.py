"""Tests for the MCP Server module."""

from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient


def test_create_search_web_tool() -> None:
    """Test that _create_search_web_tool returns a valid tool definition."""
    from mcp_server import _create_search_web_tool

    tool = _create_search_web_tool(tavily_api_key="test-key", max_results=5)
    assert tool.name == "search_web"
    assert "Tavily" in tool.description
    assert tool.inputSchema["type"] == "object"
    assert "query" in tool.inputSchema["properties"]
    assert "query" in tool.inputSchema["required"]


def test_create_mcp_app() -> None:
    """Test that create_mcp_app returns a Starlette app."""
    from mcp_server import create_mcp_app

    app = create_mcp_app(
        tavily_api_key="tvly-fake-test-key",
        max_results=3,
        route="/mcp",
    )
    # Starlette app has routes attribute
    assert hasattr(app, "routes")
    # Should have /mcp and /health routes
    route_paths = [getattr(r, "path", None) for r in app.routes]
    assert "/mcp" in route_paths
    assert "/health" in route_paths


def test_health_response() -> None:
    """Test the health check response helper."""
    from mcp_server import _health_response

    response = _health_response()
    assert response.status_code == 200
    data = response.body.decode()
    assert '"status"' in data
    assert "ok" in data
    assert "devmate-mcp-server" in data


async def test_health_endpoint_returns_ok() -> None:
    """Test GET /health returns 200 with status ok."""
    from mcp_server import create_mcp_app

    app = create_mcp_app(tavily_api_key="fake-key", route="/mcp")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "devmate-mcp-server"


async def test_mcp_endpoint_responds() -> None:
    """Test that /mcp/ endpoint responds to POST requests.

    Note: Streamable HTTP transport requires task group initialization
    which is only available when running via uvicorn. This test uses
    a real server (same pattern as integration tests).
    """
    import asyncio
    import socket

    import uvicorn

    from mcp_server import create_mcp_app

    app = create_mcp_app(tavily_api_key="fake-key", route="/mcp")

    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)

    async def _run_server():
        await server.serve()

    task = asyncio.create_task(_run_server())

    # Wait for server to be ready
    for _ in range(20):
        try:
            async with AsyncClient() as client:
                r = await client.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(0.25)

    try:
        async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            response = await client.post(
                "/mcp/",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.1.0"},
                    },
                },
            )
        assert response.status_code in (200, 202)
    finally:
        server.should_exit = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_execute_search_web_success() -> None:
    """Test _execute_search_web with mocked TavilyClient."""
    from mcp_server import _execute_search_web

    mock_response = {
        "answer": "Test answer",
        "results": [
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Test content",
            }
        ],
    }

    with patch("tavily.TavilyClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.search.return_value = mock_response
        mock_cls.return_value = mock_client

        result = await _execute_search_web(
            query="test query",
            tavily_api_key="fake-key",
            max_results=3,
        )

    assert len(result) == 1
    assert "Test answer" in result[0].text
    mock_client.search.assert_called_once()


async def test_execute_search_web_error() -> None:
    """Test _execute_search_web handles API errors gracefully."""
    from mcp_server import _execute_search_web

    with patch("tavily.TavilyClient") as mock_cls:
        mock_cls.side_effect = Exception("API error")

        result = await _execute_search_web(
            query="test query",
            tavily_api_key="fake-key",
        )

    assert len(result) == 1
    assert "Search error" in result[0].text


async def test_execute_search_web_empty_results() -> None:
    """Test _execute_search_web handles empty results."""
    from mcp_server import _execute_search_web

    with patch("tavily.TavilyClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.search.return_value = {"answer": "", "results": []}
        mock_cls.return_value = mock_client

        result = await _execute_search_web(
            query="test query",
            tavily_api_key="fake-key",
        )

    assert len(result) == 1
    assert "No results found" in result[0].text


def test_server_main_logs_error_on_missing_config() -> None:
    """Test server.main() handles FileNotFoundError."""
    from mcp_server.server import main

    with patch("mcp_server.server.load_config") as mock_load:
        mock_load.side_effect = FileNotFoundError("config.toml not found")
        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_with(1)
