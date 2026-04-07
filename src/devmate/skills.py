"""Skills system for DevMate.

Scans .skills/ directory for markdown files, parses frontmatter,
and registers skills as LangChain tools.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Skill:
    """Represents a registered skill."""

    name: str
    description: str
    trigger_keywords: list[str] = field(default_factory=list)
    content: str = ""
    source_file: Path | None = None


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

        # Parse list values
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
        return content[match.end() :]
    return content


class SkillsManager:
    """Manages loading, registering, and querying skills."""

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

        Returns:
            Number of skills loaded.
        """
        if not self._skills_dir.exists():
            logger.warning("Skills directory does not exist: %s", self._skills_dir)
            return 0

        md_files = list(self._skills_dir.glob("**/*.md"))
        logger.info("Found %d skill files in %s", len(md_files), self._skills_dir)

        count = 0
        for md_file in md_files:
            try:
                skill = self._load_skill(md_file)
                if skill:
                    self._skills[skill.name] = skill
                    count += 1
                    logger.info("Loaded skill: %s", skill.name)
            except Exception as exc:
                logger.error("Failed to load skill %s: %s", md_file, exc)

        logger.info("Total skills loaded: %d", count)
        return count

    def _load_skill(self, file_path: Path) -> Skill | None:
        """Load a single skill from a markdown file.

        Args:
            file_path: Path to the skill markdown file.

        Returns:
            A Skill instance, or None if parsing fails.
        """
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return None

        frontmatter = _parse_frontmatter(content)
        name = frontmatter.get("name", file_path.stem)
        description = frontmatter.get(
            "description", f"Skill loaded from {file_path.name}"
        )
        trigger_keywords = frontmatter.get("trigger_keywords", [])
        body = _extract_body(content)

        return Skill(
            name=name,
            description=description,
            trigger_keywords=trigger_keywords,
            content=body,
            source_file=file_path,
        )

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: The skill name.

        Returns:
            The Skill instance, or None if not found.
        """
        return self._skills.get(name)

    def find_matching_skills(self, query: str) -> list[Skill]:
        """Find skills matching a query based on trigger keywords.

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

    def save_skill(self, skill: Skill) -> None:
        """Save a skill to the skills directory.

        Args:
            skill: The Skill instance to save.
        """
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._skills_dir / f"{skill.name}.md"

        # Build frontmatter
        lines = ["---"]
        lines.append(f'name: "{skill.name}"')
        lines.append(f'description: "{skill.description}"')
        if skill.trigger_keywords:
            keywords = ", ".join(f'"{kw}"' for kw in skill.trigger_keywords)
            lines.append(f"trigger_keywords: [{keywords}]")
        lines.append("---")
        lines.append("")
        lines.append(skill.content)

        file_path.write_text("\n".join(lines), encoding="utf-8")
        self._skills[skill.name] = skill
        skill.source_file = file_path
        logger.info("Saved skill: %s -> %s", skill.name, file_path)

    def get_all_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def create_tools(self) -> list[Any]:
        """Create LangChain tools from all loaded skills.

        Returns:
            A list of LangChain tool functions.
        """
        tools: list[Any] = []

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
                available = ", ".join(s.name for s in self._skills.values())
                return f"No matching skills found. Available skills: {available}"

            results: list[str] = []
            for skill in matching:
                results.append(
                    f"--- Skill: {skill.name} ---\n"
                    f"Description: {skill.description}\n"
                    f"Content:\n{skill.content}"
                )
            return "\n\n".join(results)

        tools.append(query_skills)
        return tools
