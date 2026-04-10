"""Skills system for DevMate.

Implements Agent Skills using LangChain's ``StructuredTool`` pattern.

Each skill is stored as a directory containing a ``SKILL.md`` file with
YAML frontmatter (name, description) and body content.  The manager
converts loaded skills into LangChain ``StructuredTool`` instances so
they integrate natively with LangChain agents and tool-calling.

This follows the LangChain recommended approach for dynamic tool
creation — see:
https://python.langchain.com/docs/concepts/tools/

Skill directory layout::

    .skills/
      my-skill/
        SKILL.md        # frontmatter + instructions
        scripts/        # (optional) helper scripts
        references/     # (optional) reference files

Compatibility: compatible with the `anthropics/skills
<https://github.com/anthropics/skills>`_ repository format so those
skills can be dropped into the skills directory for testing.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Pydantic schemas for LangChain StructuredTool input validation
# ---------------------------------------------------------------------------


class SkillLoadInput(BaseModel):
    """Input schema for loading a skill by name."""

    name: str = Field(description="The exact name of the skill to load.")


class SkillSaveInput(BaseModel):
    """Input schema for saving a new skill."""

    name: str = Field(description="Short identifier (used as folder name).")
    description: str = Field(description="Brief description of what the skill does.")
    content: str = Field(description="Full instruction content (markdown) for the skill.")


class SkillQueryInput(BaseModel):
    """Input schema for querying / searching skills."""

    query: str = Field(description="Description of what you need help with.")


class SkillExecuteInput(BaseModel):
    """Input schema for executing a skill directly as a tool.

    Each registered skill is also exposed as an individual StructuredTool
    whose arguments are defined by the skill's own ``parameters`` frontmatter.
    """

    arguments: str = Field(
        default="",
        description=(
            "JSON string of key-value arguments for the skill. "
            "Pass an empty string if no arguments are needed."
        ),
    )


# ---------------------------------------------------------------------------
# Skill data model
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """Represents a registered skill."""

    name: str
    description: str
    content: str = ""
    base_dir: Path | None = None
    trigger_keywords: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def source_file(self) -> Path | None:
        """Return the SKILL.md path."""
        if self.base_dir is not None:
            return self.base_dir / "SKILL.md"
        return None

    @source_file.setter
    def source_file(self, value: Path | None) -> None:
        """Set base_dir from source_file."""
        if value is not None:
            self.base_dir = value.parent
        else:
            self.base_dir = None

    def get_detail(self) -> str:
        """Return the skill's full content with resolved paths."""
        base_dir_str = str(self.base_dir) if self.base_dir else ""
        content = self.content.replace("<SkillDir>", base_dir_str)
        content = content.replace("${SKILL_DIR}", base_dir_str)
        if base_dir_str:
            return f"Base directory for this skill: {base_dir_str}\n\n{content}"
        return content


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_skill(md_path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill instance.

    Extracts YAML frontmatter fields and the body content.  Falls back
    to using the parent directory name as the skill name and the first
    heading line as the description.

    Args:
        md_path: Path to the SKILL.md file.

    Returns:
        A Skill instance, or None if the file is empty or missing.
    """
    if not md_path.exists():
        logger.warning("Skill file not found: %s", md_path)
        return None

    raw = md_path.read_text(encoding="utf-8")
    if not raw.strip():
        logger.warning("Skill file is empty: %s", md_path)
        return None

    frontmatter = _parse_frontmatter(raw)
    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")
    trigger_keywords = frontmatter.get("trigger_keywords", [])
    parameters = frontmatter.get("parameters", {})
    body = _extract_body(raw)

    if not name:
        name = md_path.parent.name

    if not description and body:
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped:
                description = re.sub(r"^#+\s*", "", stripped).strip()
                break

    return Skill(
        name=name,
        description=description,
        content=body,
        base_dir=md_path.parent,
        trigger_keywords=trigger_keywords if isinstance(trigger_keywords, list) else [],
        parameters=parameters if isinstance(parameters, dict) else {},
    )


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML-like frontmatter from markdown content."""
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}

    frontmatter_str = match.group(1)
    result: dict[str, Any] = {}
    for line in frontmatter_str.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if inner:
                result[key] = [item.strip().strip("\"'") for item in inner.split(",")]
            else:
                result[key] = []
        else:
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            result[key] = value

    return result


def _extract_body(content: str) -> str:
    """Extract the markdown body (after frontmatter)."""
    match = _FRONTMATTER_PATTERN.match(content)
    if match:
        return content[match.end() :].strip()
    return content.strip()


# ---------------------------------------------------------------------------
# SkillsManager — loads skills, exposes them as LangChain StructuredTools
# ---------------------------------------------------------------------------


class SkillsManager:
    """Manages loading, querying, and saving folder-based skills.

    Each skill is a directory containing a ``SKILL.md`` file.  Loaded
    skills are converted into LangChain ``StructuredTool`` instances via
    :meth:`create_tools`, following the LangChain recommended pattern for
    dynamic tool registration.

    Compatible with the `anthropics/skills
    <https://github.com/anthropics/skills>`_ repository format.
    """

    def __init__(self, skills_dir: str | Path = ".skills") -> None:
        """Initialize the skills manager.

        Args:
            skills_dir: Path to the skills directory.
        """
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        logger.info("Skills manager initialized (dir=%s)", self._skills_dir)

    def load_skills(self) -> int:
        """Load all skills from the skills directory.

        Scans for subdirectories containing SKILL.md files.

        Returns:
            Number of skills loaded.
        """
        if not self._skills_dir.exists():
            logger.warning("Skills directory does not exist: %s", self._skills_dir)
            return 0

        if not self._skills_dir.is_dir():
            logger.warning("Skills path is not a directory: %s", self._skills_dir)
            return 0

        count = 0
        for entry in sorted(self._skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            try:
                skill = parse_skill(skill_md)
                if skill and skill.name not in self._skills:
                    self._skills[skill.name] = skill
                    count += 1
                    logger.info("Loaded skill: %s", skill.name)
            except Exception as exc:
                logger.error("Failed to load skill from %s: %s", entry, exc)

        logger.info("Total skills loaded: %d", count)
        return count

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by exact name."""
        return self._skills.get(name)

    def execute_skill(self, name: str) -> str:
        """Retrieve a skill's full content by name."""
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills.keys()))
            return f"Skill '{name}' not found. Available skills: {available}"
        return skill.get_detail()

    def get_skill_meta(self) -> str:
        """Return XML-formatted summary of all loaded skills.

        Intended for injection into the System Prompt.
        """
        if not self._skills:
            return ""

        lines = ["<available_skills>"]
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name}</name>")
            lines.append("    <description>")
            lines.append(f"      {skill.description}")
            lines.append("    </description>")
            lines.append("    <location>user</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_all_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def save_skill(self, name: str, description: str, content: str) -> Skill:
        """Save a new skill as a folder with SKILL.md."""
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        md_lines = ["---"]
        md_lines.append(f'name: "{name}"')
        md_lines.append(f'description: "{description}"')
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(content)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("\n".join(md_lines), encoding="utf-8")

        skill = Skill(
            name=name,
            description=description,
            content=content,
            base_dir=skill_dir,
        )
        self._skills[name] = skill
        logger.info("Saved skill: %s -> %s", name, skill_dir)
        return skill

    def find_matching_skills(self, query: str) -> list[Skill]:
        """Find skills matching a query based on trigger keywords and name/description."""
        query_lower = query.lower()
        matches: list[tuple[Skill, int]] = []

        for skill in self._skills.values():
            score = 0
            # Keyword matching
            for keyword in skill.trigger_keywords:
                if keyword.lower() in query_lower:
                    score += 2
            # Name / description matching (fuzzy)
            if skill.name.lower() in query_lower:
                score += 3
            for word in query_lower.split():
                if word in skill.description.lower():
                    score += 1
            if score > 0:
                matches.append((skill, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in matches]

    # ------------------------------------------------------------------
    # LangChain StructuredTool creation (recommended pattern)
    # ------------------------------------------------------------------

    def create_tools(self) -> list[StructuredTool]:
        """Create LangChain ``StructuredTool`` instances for skill interaction.

        This follows the LangChain recommended pattern for dynamic tool
        creation using ``StructuredTool`` with Pydantic input schemas.

        Returns:
            A list of ``StructuredTool`` instances:

            - ``skill_load`` — load a skill's full instructions by name.
            - ``skill_save`` — save a new skill to the skills directory.
            - ``skill_query`` — search available skills by query.
            - One ``use_<skill_name>`` tool per loaded skill — executes
              (retrieves) the skill content directly, so the agent can
              call individual skills as first-class tools.
        """
        tools: list[StructuredTool] = []

        # --- meta-tools: load, save, query ---

        def _load_skill(name: str) -> str:
            return self.execute_skill(name)

        tools.append(
            StructuredTool.from_function(
                func=_load_skill,
                name="skill_load",
                description=(
                    "Load and execute a skill by its exact name to get its "
                    "full instruction content. Use this when you need the "
                    "detailed instructions for a specific skill."
                ),
                args_schema=SkillLoadInput,
            )
        )

        def _save_skill(name: str, description: str, content: str) -> str:
            saved = self.save_skill(name, description, content)
            return f"Skill '{saved.name}' saved successfully to {saved.base_dir}"

        tools.append(
            StructuredTool.from_function(
                func=_save_skill,
                name="skill_save",
                description=(
                    "Save a new skill with the given name, description, and "
                    "content. The skill will be stored as a folder with a "
                    "SKILL.md file and can be loaded later."
                ),
                args_schema=SkillSaveInput,
            )
        )

        def _query_skills(query: str) -> str:
            matching = self.find_matching_skills(query)
            if not matching:
                available = ", ".join(sorted(self._skills.keys()))
                return f"No matching skills found. Available skills: {available}"

            results: list[str] = []
            for s in matching:
                results.append(
                    f"--- Skill: {s.name} ---\n"
                    f"Description: {s.description}\n"
                    f"Content:\n{s.content}"
                )
            return "\n\n".join(results)

        tools.append(
            StructuredTool.from_function(
                func=_query_skills,
                name="skill_query",
                description=(
                    "Search available skills for relevant knowledge and "
                    "patterns. Skills contain reusable knowledge, code "
                    "patterns, and best practices. Use this to find skills "
                    "that match your current task."
                ),
                args_schema=SkillQueryInput,
            )
        )

        # --- per-skill tools: each loaded skill becomes a StructuredTool ---

        for skill in self._skills.values():
            safe_name = f"use_{re.sub(r'[^a-zA-Z0-9_]', '_', skill.name)}"

            def _make_executor(
                skill_name: str,
            ) -> Any:
                """Create a closure capturing the skill name."""

                def _execute(arguments: str = "") -> str:
                    return self.execute_skill(skill_name)

                return _execute

            tools.append(
                StructuredTool.from_function(
                    func=_make_executor(skill.name),
                    name=safe_name,
                    description=(
                        f"Execute the '{skill.name}' skill. "
                        f"{skill.description}"
                    ),
                    args_schema=SkillExecuteInput,
                )
            )

        logger.info(
            "Created %d StructuredTool instances from skills", len(tools)
        )
        return tools
