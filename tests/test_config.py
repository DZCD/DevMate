"""Tests for the DevMate configuration module."""

import os
import tempfile

import pytest

from devmate.config import get_model_config, load_config


def test_load_config_with_valid_file() -> None:
    """Test loading a valid TOML configuration file."""
    config_content = """
[model]
base_url = "https://example.com/api"
api_key = "test-key"
model_name = "test-model"

[search]
tavily_api_key = "tvly-test"
max_results = 3

[langsmith]
enabled = false

[skills]
directory = ".skills"

[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"

[mcp_server]
host = "0.0.0.0"
port = 8001
route = "/mcp"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        f.flush()
        config = load_config(f.name)

    assert config is not None
    assert config["model"]["base_url"] == "https://example.com/api"
    assert config["model"]["api_key"] == "test-key"
    assert config["search"]["max_results"] == 3
    assert config["mcp_server"]["port"] == 8001

    os.unlink(f.name)


def test_load_config_missing_file() -> None:
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.toml")


def test_get_model_config() -> None:
    """Test extracting model configuration."""
    config = {
        "model": {
            "base_url": "https://api.example.com",
            "model_name": "test-llm",
        },
        "search": {},
    }
    model_config = get_model_config(config)
    assert model_config["base_url"] == "https://api.example.com"
    assert model_config["model_name"] == "test-llm"
