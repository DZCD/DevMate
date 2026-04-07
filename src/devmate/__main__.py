"""CLI entry point for DevMate.

Provides commands:
- devmate init: Initialize document index
- devmate chat: Start interactive chat session
- devmate run "prompt": Execute a single task
- devmate serve: Start the MCP server
"""

import asyncio
import logging
import sys

import click

from devmate import __version__

logger = logging.getLogger("devmate")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


@click.group()
@click.version_option(version=__version__, prog_name="devmate")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DevMate - AI-powered development assistant."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config.toml file.",
)
@click.pass_context
def init(ctx: click.Context, config: str | None) -> None:
    """Initialize the document index for RAG."""
    from devmate.config import get_rag_config, load_config
    from devmate.rag import RAGEngine

    click.echo("Initializing DevMate document index...")
    try:
        cfg = load_config(config)
        rag_config = get_rag_config(cfg)
        engine = RAGEngine(
            persist_directory=rag_config.get("chroma_persist_directory", ".chroma_db"),
            chunk_size=rag_config.get("chunk_size", 1000),
            chunk_overlap=rag_config.get("chunk_overlap", 200),
        )
        count = engine.ingest_documents(rag_config.get("docs_directory", "docs"))
        click.echo(f"Successfully indexed {count} document chunks.")
        click.echo(f"Total documents in index: {engine.get_doc_count()}")
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Init failed: %s", exc, exc_info=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config.toml file.",
)
@click.option(
    "--workspace",
    "-w",
    default=None,
    help="Workspace directory path.",
)
@click.pass_context
def chat(ctx: click.Context, config: str | None, workspace: str | None) -> None:
    """Start an interactive chat session."""
    from devmate.agent import create_agent_func

    click.echo("Starting DevMate chat session...")
    click.echo("Type 'exit', 'quit', or press Ctrl+C to end.")

    agent = create_agent_func(config_path=config, workspace=workspace)

    async def _run() -> None:
        try:
            await agent.chat_loop()
        finally:
            await agent.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
    except Exception as exc:
        logger.error("Chat failed: %s", exc, exc_info=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config.toml file.",
)
@click.option(
    "--workspace",
    "-w",
    default=None,
    help="Workspace directory path.",
)
@click.pass_context
def run(
    ctx: click.Context, prompt: str, config: str | None, workspace: str | None
) -> None:
    """Execute a single task with the given prompt."""
    from devmate.agent import create_agent_func

    agent = create_agent_func(config_path=config, workspace=workspace)

    async def _run() -> str:
        try:
            result = await agent.run(prompt)
            return result
        finally:
            await agent.cleanup()

    try:
        response = asyncio.run(_run())
        click.echo(response)
    except Exception as exc:
        logger.error("Run failed: %s", exc, exc_info=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config.toml file.",
)
@click.option(
    "--host",
    "-h",
    default=None,
    help="Host to bind to.",
)
@click.option(
    "--port",
    "-p",
    default=None,
    type=int,
    help="Port to listen on.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    config: str | None,
    host: str | None,
    port: int | None,
) -> None:
    """Start the MCP server."""
    import uvicorn

    from devmate.config import get_mcp_server_config, get_search_config, load_config
    from mcp_server import create_mcp_app

    click.echo("Starting DevMate MCP Server...")
    try:
        cfg = load_config(config)
        search_config = get_search_config(cfg)
        mcp_config = get_mcp_server_config(cfg)

        tavily_api_key = search_config.get("tavily_api_key", "")
        max_results = search_config.get("max_results", 5)
        bind_host = host or mcp_config.get("host", "0.0.0.0")
        bind_port = port or mcp_config.get("port", 8001)
        route = mcp_config.get("route", "/mcp")

        if not tavily_api_key or tavily_api_key.startswith("YOUR_"):
            click.echo(
                "Warning: Tavily API key not configured. "
                "Set 'tavily_api_key' in config.toml [search] section.",
                err=True,
            )

        app = create_mcp_app(
            tavily_api_key=tavily_api_key,
            max_results=max_results,
            route=route,
        )

        click.echo(f"MCP Server starting on http://{bind_host}:{bind_port}{route}")
        uvicorn.run(app, host=bind_host, port=bind_port)

    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Serve failed: %s", exc, exc_info=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
