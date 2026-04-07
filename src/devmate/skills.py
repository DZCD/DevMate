"""Skills system for DevMate.

Scans skills directories for folder-based skills, each containing a
SKILL.md file with YAML frontmatter (name, description) and body content.
Provides functions for meta listing, detail retrieval, saving, and parsing.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Skill:
    """Represents a registered skill."""

    name: str
    description: str
    content: str = ""
    base_dir: Path | None = None
    trigger_keywords: list[str] = field(default_factory=list)

    # Backward compatibility: alias for base_dir
    @property
    def source_file(self) -> Path | None:
        """Return the SKILL.md path for backward compatibility."""
        if self.base_dir is not None:
            return self.base_dir / "SKILL.md"
        return None

    @source_file.setter
    def source_file(self, value: Path | None) -> None:
        """Set base_dir from source_file for backward compatibility."""
        if value is not None:
            self.base_dir = value.parent
        else:
            self.base_dir = None

    def get_detail(self) -> str:
        """Return the skill's full content with base directory info."""
        base_dir_str = str(self.base_dir) if self.base_dir else ""
        # Replace placeholder tokens
        content = self.content.replace("<SkillDir>", base_dir_str)
        content = content.replace("${SKILL_DIR}", base_dir_str)
        if base_dir_str:
            return f"Base directory for this skill: {base_dir_str}\n\n{content}"
        return content


def parse_skill(md_path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill instance.

    Extracts YAML frontmatter fields (name, description) and the body
    content. Falls back to using the parent directory name as the skill
    name and the first heading line as the description.

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
    body = _extract_body(raw)

    # Fallback: use parent directory name as skill name
    if not name:
        name = md_path.parent.name

    # Fallback: use first non-empty line (heading) as description
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
    )


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML-like frontmatter from markdown content.

    Args:
        content: The full markdown file content.

    Returns:
        A dictionary of frontmatter fields.
    """
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

        # Parse list values (YAML inline array)
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if inner:
                result[key] = [item.strip().strip("\"'") for item in inner.split(",")]
            else:
                result[key] = []
        else:
            # Remove surrounding quotes
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


class SkillsManager:
    """Manages loading, querying, and saving folder-based skills.

    Each skill is a directory containing a SKILL.md file.
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
        """Get a skill by exact name.

        Args:
            name: The skill name.

        Returns:
            The Skill instance, or None if not found.
        """
        return self._skills.get(name)

    def execute_skill(self, name: str) -> str:
        """Execute (retrieve) a skill's full content by name.

        Args:
            name: The skill name.

        Returns:
            The skill's full detail content, or an error message if not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills.keys()))
            return f"Skill '{name}' not found. Available skills: {available}"
        return skill.get_detail()

    def get_skill_meta(self) -> str:
        """Return XML-formatted summary of all loaded skills.

        This is intended for injection into the System Prompt so the
        agent knows which skills are available.

        Returns:
            XML string listing all skills with name and description.
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
        """Save a new skill as a folder with SKILL.md.

        Args:
            name: The skill name (used as folder name and frontmatter name).
            description: A brief description of the skill.
            content: The skill body content (markdown instructions).

        Returns:
            The newly created Skill instance.
        """
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build SKILL.md with frontmatter
        lines = ["---"]
        lines.append(f'name: "{name}"')
        lines.append(f'description: "{description}"')
        lines.append("---")
        lines.append("")
        lines.append(content)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("\n".join(lines), encoding="utf-8")

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
        """Find skills matching a query based on trigger keywords.

        .. deprecated::
            This method is kept for backward compatibility but is no longer
            the primary way to discover skills. Use get_skill_meta() and
            execute_skill() instead.

        Args:
            query: The user query to match against.

        Returns:
            A list of matching skills, sorted by number of keyword matches.
        """
        query_lower = query.lower()
        matches: list[tuple[Skill, int]] = []

        for skill in self._skills.values():
            score = 0
            for keyword in skill.trigger_keywords:
                if keyword.lower() in query_lower:
                    score += 1
            if score > 0:
                matches.append((skill, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in matches]

    def create_tools(self) -> list[Any]:
        """Create LangChain tools for skill interaction.

        Returns:
            A list of LangChain tool functions.
        """
        from langchain_core.tools import tool

        tools: list[Any] = []

        @tool
        def skill(name: str) -> str:
            """Load and execute a skill by its exact name to get its full
            instruction content. Use this when you need the detailed
            instructions for a specific skill.

            Args:
                name: The exact name of the skill to load.
            """
            return self.execute_skill(name)

        @tool
        def save_skill(name: str, description: str, content: str) -> str:
            """Save a new skill with the given name, description, and content.
            The skill will be stored as a folder with a SKILL.md file.

            Args:
                name: A short identifier for the skill (used as folder name).
                description: A brief description of what the skill does.
                content: The full instruction content (markdown) for the skill.
            """
            skill = self.save_skill(name, description, content)
            return f"Skill '{skill.name}' saved successfully to {skill.base_dir}"

        @tool
        def query_skills(query: str) -> str:
            """Search available skills for relevant knowledge and patterns.

            Skills contain reusable knowledge, code patterns, and best practices.
            Use this tool to find skills that match your current task.

            Args:
                query: A description of what you need help with.
            """
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

        tools.extend([skill, save_skill, query_skills])
        return tools
