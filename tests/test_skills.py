"""Tests for the Skills module."""

import tempfile
from pathlib import Path

from devmate.skills import (
    Skill,
    SkillsManager,
    _extract_body,
    _parse_frontmatter,
    parse_skill,
)

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

SAMPLE_SKILL_DIR = "test_skill/SKILL.md"


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


def test_parse_skill_function() -> None:
    """Test the parse_skill standalone function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        skill = parse_skill(skill_dir / "SKILL.md")
        assert skill is not None
        assert skill.name == "test_skill"
        assert skill.description == "A test skill for unit testing"
        assert "# Test Skill" in skill.content
        assert skill.base_dir == skill_dir


def test_parse_skill_missing_file() -> None:
    """Test parse_skill returns None for missing file."""
    skill = parse_skill(Path("/nonexistent/SKILL.md"))
    assert skill is None


def test_parse_skill_fallback_name() -> None:
    """Test parse_skill uses directory name as fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "my_custom_skill"
        skill_dir.mkdir()
        content = "# Custom Skill\n\nSome content."
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        skill = parse_skill(skill_dir / "SKILL.md")
        assert skill is not None
        assert skill.name == "my_custom_skill"


def test_parse_skill_fallback_description() -> None:
    """Test parse_skill uses first heading as fallback description."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "my_skill"
        skill_dir.mkdir()
        content = "# My Skill Description\n\nBody content here."
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        skill = parse_skill(skill_dir / "SKILL.md")
        assert skill is not None
        assert skill.description == "My Skill Description"


def test_skill_get_detail() -> None:
    """Test Skill.get_detail includes base directory."""
    skill = Skill(
        name="test",
        description="desc",
        content="Instructions here.",
        base_dir=Path("/tmp/skills/test"),
    )
    detail = skill.get_detail()
    assert "Base directory for this skill: /tmp/skills/test" in detail
    assert "Instructions here." in detail


def test_skill_source_file_backward_compat() -> None:
    """Test Skill.source_file property for backward compatibility."""
    skill = Skill(
        name="test",
        description="desc",
        base_dir=Path("/tmp/skills/test"),
    )
    assert skill.source_file == Path("/tmp/skills/test/SKILL.md")


def test_skill_source_file_setter() -> None:
    """Test Skill.source_file setter updates base_dir."""
    skill = Skill(name="test", description="desc")
    skill.source_file = Path("/tmp/skills/test/SKILL.md")
    assert skill.base_dir == Path("/tmp/skills/test")


def test_skills_manager_load() -> None:
    """Test loading skills from directory with folder structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        # Create a skill folder with SKILL.md
        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        count = manager.load_skills()
        assert count == 1

        skill = manager.get_skill("test_skill")
        assert skill is not None
        assert skill.description == "A test skill for unit testing"
        assert "test" in skill.trigger_keywords


def test_skills_manager_load_ignores_md_files() -> None:
    """Test that loose .md files are ignored (only folders with SKILL.md)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        # Create a loose .md file (old format, should be ignored)
        (skills_dir / "old_skill.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        count = manager.load_skills()
        assert count == 0


def test_skills_manager_execute_skill() -> None:
    """Test execute_skill returns full content by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        result = manager.execute_skill("test_skill")
        assert "Test Skill" in result
        assert "Base directory for this skill:" in result

        # Non-existent skill
        result = manager.execute_skill("nonexistent")
        assert "not found" in result


def test_skills_manager_get_skill_meta() -> None:
    """Test get_skill_meta returns XML format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "alpha"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Alpha skill\n---\n# Alpha\n",
            encoding="utf-8",
        )

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        meta = manager.get_skill_meta()
        assert "<available_skills>" in meta
        assert "</available_skills>" in meta
        assert "<name>alpha</name>" in meta
        assert "<description>" in meta
        assert "Alpha skill" in meta


def test_skills_manager_get_skill_meta_empty() -> None:
    """Test get_skill_meta returns empty string when no skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SkillsManager(skills_dir=tmpdir)
        meta = manager.get_skill_meta()
        assert meta == ""


def test_skills_manager_save() -> None:
    """Test saving a new skill creates folder with SKILL.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        manager = SkillsManager(skills_dir=skills_dir)
        manager.save_skill(
            name="new_skill",
            description="A newly created skill",
            content="# New Skill\n\nContent here.",
        )

        # Verify folder structure
        skill_folder = skills_dir / "new_skill"
        assert skill_folder.is_dir()
        assert (skill_folder / "SKILL.md").exists()

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


def test_skills_manager_find_matching() -> None:
    """Test finding skills by keyword match (deprecated but still works)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        matches = manager.find_matching_skills("I need a mock for testing")
        assert len(matches) == 1
        assert matches[0].name == "test_skill"

        no_match = manager.find_matching_skills("something unrelated")
        assert len(no_match) == 0


def test_skills_create_tools() -> None:
    """Test that create_tools returns expected tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_CONTENT, encoding="utf-8")

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()
        tools = manager.create_tools()

        tool_names = [t.name for t in tools]
        assert "skill" in tool_names
        assert "save_skill" in tool_names
        assert "query_skills" in tool_names


def test_skill_tool_invocation() -> None:
    """Test the skill tool returns correct content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "demo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: Demo skill\n---\n# Demo\nHello.",
            encoding="utf-8",
        )

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()
        tools = manager.create_tools()

        skill_tool = next(t for t in tools if t.name == "skill")
        result = skill_tool.invoke({"name": "demo"})
        assert "Hello" in result


def test_save_skill_tool_invocation() -> None:
    """Test the save_skill tool creates a new skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".skills"
        skills_dir.mkdir()

        manager = SkillsManager(skills_dir=skills_dir)
        tools = manager.create_tools()

        save_tool = next(t for t in tools if t.name == "save_skill")
        result = save_tool.invoke(
            {
                "name": "tool_skill",
                "description": "Created via tool",
                "content": "# Tool Skill\nContent.",
            }
        )
        assert "saved successfully" in result

        # Verify the skill was actually saved
        assert (skills_dir / "tool_skill" / "SKILL.md").exists()
