"""Tests for the Skills module."""

import tempfile
from pathlib import Path

from devmate.skills import SkillsManager, _extract_body, _parse_frontmatter

SAMPLE_SKILL_CONTENT = """---
name: "test_skill"
description: "A test skill for unit testing"
trigger_keywords: ["test", "mock", "stub"]
---

# Test Skill

This is a test skill content.

## Usage

Use this skill for testing.
"""

SAMPLE_NO_FRONTMATTER = """# Plain Document

This document has no frontmatter.
"""


def test_parse_frontmatter() -> None:
    """Test frontmatter parsing."""
    result = _parse_frontmatter(SAMPLE_SKILL_CONTENT)
    assert result["name"] == "test_skill"
    assert result["description"] == "A test skill for unit testing"
    assert result["trigger_keywords"] == ["test", "mock", "stub"]


def test_parse_frontmatter_none() -> None:
    """Test frontmatter parsing with no frontmatter."""
    result = _parse_frontmatter(SAMPLE_NO_FRONTMATTER)
    assert result == {}


def test_extract_body() -> None:
    """Test body extraction."""
    body = _extract_body(SAMPLE_SKILL_CONTENT)
    assert "# Test Skill" in body
    assert "name:" not in body
    assert "Use this skill for testing." in body


def test_skills_manager_load() -> None:
    """Test loading skills from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()
        (skills_dir / "test.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        count = manager.load_skills()
        assert count == 1

        skill = manager.get_skill("test_skill")
        assert skill is not None
        assert skill.description == "A test skill for unit testing"
        assert "test" in skill.trigger_keywords


def test_skills_manager_find_matching() -> None:
    """Test finding skills by keyword match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()
        (skills_dir / "test.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        matches = manager.find_matching_skills("I need a mock for testing")
        assert len(matches) == 1
        assert matches[0].name == "test_skill"

        no_match = manager.find_matching_skills("something unrelated")
        assert len(no_match) == 0


def test_skills_manager_save() -> None:
    """Test saving a skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        manager = SkillsManager(skills_dir=skills_dir)

        from devmate.skills import Skill

        skill = Skill(
            name="new_skill",
            description="A newly created skill",
            trigger_keywords=["new", "create"],
            content="# New Skill\n\nContent here.",
        )
        manager.save_skill(skill)

        # Verify file was created
        saved_file = skills_dir / "new_skill.md"
        assert saved_file.exists()

        # Reload and verify
        manager2 = SkillsManager(skills_dir=skills_dir)
        manager2.load_skills()
        retrieved = manager2.get_skill("new_skill")
        assert retrieved is not None
        assert retrieved.description == "A newly created skill"


def test_skills_manager_empty_directory() -> None:
    """Test loading from nonexistent directory."""
    manager = SkillsManager(skills_dir="/nonexistent")
    count = manager.load_skills()
    assert count == 0


def test_skills_create_tools() -> None:
    """Test that create_tools returns a list of tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()
        (skills_dir / "test.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()
        tools = manager.create_tools()
        assert len(tools) >= 1
        assert tools[0].name == "query_skills"
