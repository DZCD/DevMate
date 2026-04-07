"""Shared test helpers for DevMate tests.

This module provides reusable test utilities that are imported by
individual test files.  Unlike conftest.py (which pytest auto-discovers
for fixtures), this is a regular Python module that must be imported
explicitly via ``from tests.helpers import ...``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def write_minimal_config(tmp_path: Path, skills_dir: str | Path | None = None) -> Path:
    """Create a minimal config.toml for testing.

    Args:
        tmp_path: Temporary directory to write the config into.
        skills_dir: Optional path to override the default skills directory.

    Returns:
        Path to the written config file.
    """
    if skills_dir is not None:
        skills_line = f'directory = "{skills_dir}"'
    else:
        skills_line = 'directory = ".skills"'

    config_content = f"""
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
{skills_line}

[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"
"""
    config_file = tmp_path / "config.toml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


async def build_agent(
    tmp_path: Path,
    mock_llm_responses: list | None = None,
) -> Any:
    """Build a fully initialized DevMateAgent with a mock LLM.

    Args:
        tmp_path: Temporary directory for config, workspace, and skills.
        mock_llm_responses: Optional list of pre-canned LLMResponse objects.
            When provided, the mock LLM's ``chat`` method returns items from
            this list in order via ``side_effect``.

    Returns:
        An initialized ``DevMateAgent`` instance.
    """
    from unittest.mock import patch

    from devmate.agent import DevMateAgent
    from devmate.llm import LLMResponse

    skills_dir = tmp_path / ".skills"
    skills_dir.mkdir(exist_ok=True)
    config_file = write_minimal_config(tmp_path, str(skills_dir))

    response_iter = iter(mock_llm_responses) if mock_llm_responses else None

    with (
        patch("devmate.agent.RAGEngine") as mock_rag_cls,
        patch("devmate.agent.SkillsManager") as mock_skills_cls,
        patch("devmate.agent.OpenAICompatibleAdapter") as mock_llm_cls,
    ):
        # RAG mock
        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance

        # Skills mock
        mock_skills_instance = MagicMock()
        mock_skills_instance.load_skills.return_value = 0
        mock_skills_instance.get_skill_meta.return_value = ""
        mock_skills_instance.create_tools.return_value = []
        mock_skills_cls.return_value = mock_skills_instance

        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
        await agent.initialize()

        # Replace the LLM's chat method if responses were provided
        if response_iter is not None:

            def _next_response(*_args, **_kwargs) -> LLMResponse:
                return next(response_iter)

            mock_llm_instance = mock_llm_cls.return_value
            mock_llm_instance.chat = AsyncMock(side_effect=_next_response)

    return agent
