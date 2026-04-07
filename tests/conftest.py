"""Shared test fixtures for DevMate tests."""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def _patch_agent_imports():
    """Patch the broken import in devmate.agent so we can test it.

    The source imports ``from langchain.agents import create_agent``
    which may not exist in langchain 1.2.x. We inject a fake into
    ``sys.modules`` before the agent module is imported.
    """
    # Create a fake langchain.agents module with create_agent
    fake_agents = types.ModuleType("langchain.agents")

    def _fake_create_agent(*args, **kwargs):
        return MagicMock()

    fake_agents.create_agent = _fake_create_agent
    fake_agents.__all__ = ["create_agent"]

    import langchain

    original_agents = getattr(langchain, "agents", None)
    langchain.agents = fake_agents
    sys.modules["langchain.agents"] = fake_agents

    # Remove cached agent module if it was already imported
    cached = sys.modules.pop("devmate.agent", None)

    yield

    # Restore
    if cached is not None:
        sys.modules["devmate.agent"] = cached
    sys.modules.pop("langchain.agents", None)
    if original_agents is not None:
        langchain.agents = original_agents
