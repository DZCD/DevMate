---
name: "example_skill"
description: "An example skill demonstrating the DevMate folder-based skill format"
---

# Example Skill

This is an example skill file that demonstrates the DevMate folder-based skill format.

## Structure

Each skill is a folder containing a `SKILL.md` file with YAML frontmatter and body content.

## Frontmatter Fields

- **name**: Unique identifier for the skill (optional, falls back to folder name)
- **description**: What this skill does (optional, falls back to first heading)

## Usage

When the agent needs guidance for a specific task, it loads the skill by name
using the `skill(name)` tool and follows the instructions in the body content.

## Best Practices

1. Keep skills focused on a single topic or pattern
2. Include code examples when relevant
3. Document edge cases and common pitfalls
4. Update skills as the project evolves
