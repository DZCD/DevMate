"""Shared test fixtures for DevMate tests.

Pytest auto-discovers this file for fixtures.  Re-exports the helpers
from ``tests.helpers`` so that all test modules can use them as fixtures
without explicit imports.
"""

from __future__ import annotations

import pytest

from tests.helpers import write_minimal_config


@pytest.fixture
def minimal_config(tmp_path):
    """Provide a Path to a minimal config.toml in *tmp_path*."""
    return write_minimal_config(tmp_path)
