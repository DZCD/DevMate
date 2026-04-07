"""Tests for the DevMate configuration module."""

import os
import tempfile

import pytest

from devmate.config import get_embedding_config, get_model_config, load_config


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

[embedding]
provider = "jina"
api_key = "YOUR_JINA_API_KEY"
model_name = "jina-embeddings-v5-text-small"
base_url = "https://api.jina.ai/v1"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        f.flush()
        config = load_config(f.name)

    assert config is not None
    assert config["model"]["base_url"] == "https://example.com/api"
    assert config["model"]["api_key"] == "test-key"
    assert config["search"]["max_results"] == 3
    assert config["embedding"]["provider"] == "jina"
    assert config["embedding"]["model_name"] == "jina-embeddings-v5-text-small"

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


def test_get_embedding_config() -> None:
    """Test extracting embedding configuration."""
    config = {
        "model": {},
        "embedding": {
            "provider": "jina",
            "api_key": "test-key",
            "model_name": "jina-embeddings-v5-text-small",
            "base_url": "https://api.jina.ai/v1",
        },
    }
    embedding_config = get_embedding_config(config)
    assert embedding_config["provider"] == "jina"
    assert embedding_config["api_key"] == "test-key"
    assert embedding_config["model_name"] == "jina-embeddings-v5-text-small"


def test_get_embedding_config_missing() -> None:
    """Test get_embedding_config returns empty dict when section absent."""
    config = {"model": {}}
    embedding_config = get_embedding_config(config)
    assert embedding_config == {}
