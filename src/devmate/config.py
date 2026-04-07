"""Configuration loader for DevMate.

Loads configuration from config.toml in the project root or a custom path.
"""

import logging
import os
from pathlib import Path
from typing import Any

import toml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.toml")


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Args:
        config_path: Path to the config file. If None, looks for
            config.toml in the current working directory.

    Returns:
        A dictionary containing all configuration sections.

    Raises:
        FileNotFoundError: If the config file does not exist.
        toml.TomlDecodeError: If the config file contains invalid TOML.
    """
    if config_path is None:
        # Try multiple locations
        candidates = [
            Path.cwd() / "config.toml",
            Path(__file__).resolve().parent.parent.parent.parent / "config.toml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        msg = "No config.toml found. Please create one from config.toml.example"
        raise FileNotFoundError(msg)

    config_path = Path(config_path)
    logger.info("Loading configuration from %s", config_path)

    config = toml.load(config_path)
    _apply_langsmith_env(config)
    return config


def _apply_langsmith_env(config: dict[str, Any]) -> None:
    """Set LangSmith environment variables from config."""
    langsmith_config = config.get("langsmith", {})

    if langsmith_config.get("enabled", False):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        api_key = langsmith_config.get("langchain_api_key", "")
        if api_key and not api_key.startswith("YOUR_"):
            os.environ["LANGCHAIN_API_KEY"] = api_key
        project_name = langsmith_config.get("project_name", "devmate")
        os.environ["LANGCHAIN_PROJECT"] = project_name
        logger.info("LangSmith tracing enabled, project: %s", project_name)
    else:
        logger.info("LangSmith tracing disabled")


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model configuration."""
    return config.get("model", {})


def get_search_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract search configuration."""
    return config.get("search", {})


def get_rag_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract RAG configuration."""
    return config.get("rag", {})


def get_skills_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract skills configuration."""
    return config.get("skills", {})


def get_mcp_server_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract MCP server configuration."""
    return config.get("mcp_server", {})
