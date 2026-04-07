"""File operation tools for DevMate.

Provides LangChain tools for file system operations:
- create_file: Create a new file
- write_file: Write content to a file
- list_directory: List directory contents
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_file_tools(workspace: str | Path | None = None) -> list[Any]:
    """Create file operation tools.

    Args:
        workspace: The workspace directory to operate in.
            If None, uses the current working directory.

    Returns:
        A list of LangChain tool functions.
    """
    if workspace is None:
        workspace = Path.cwd()
    workspace = Path(workspace)

    @tool
    def create_file(file_path: str, content: str = "", overwrite: bool = False) -> str:
        """Create a new file at the specified path.

        Creates the file (and any necessary parent directories) with the
        given content. Will not overwrite existing files unless overwrite=True.

        Args:
            file_path: Relative or absolute path for the new file.
            content: Initial content to write to the file.
            overwrite: If True, overwrite existing files. Default False.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target

        # Normalize path to stay within workspace
        try:
            target = target.resolve()
            workspace_resolved = workspace.resolve()
            if not str(target).startswith(str(workspace_resolved)):
                return f"Error: Path {target} is outside the workspace."
        except Exception as exc:
            return f"Error resolving path: {exc}"

        if target.exists() and not overwrite:
            return (
                f"Error: File already exists: {target}. Use overwrite=True to replace."
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            logger.info("Created file: %s", target)
            return f"Successfully created file: {target}"
        except Exception as exc:
            logger.error("Failed to create file %s: %s", target, exc)
            return f"Error creating file: {exc}"

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to an existing file, replacing all current content.

        Use this tool to update files with new content. The file must already exist.

        Args:
            file_path: Path to the file to write to.
            content: The new content to write.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target

        try:
            target = target.resolve()
            workspace_resolved = workspace.resolve()
            if not str(target).startswith(str(workspace_resolved)):
                return f"Error: Path {target} is outside the workspace."
        except Exception as exc:
            return f"Error resolving path: {exc}"

        if not target.exists():
            return f"Error: File does not exist: {target}. Use create_file first."

        try:
            target.write_text(content, encoding="utf-8")
            logger.info("Wrote file: %s", target)
            return f"Successfully wrote to file: {target}"
        except Exception as exc:
            logger.error("Failed to write file %s: %s", target, exc)
            return f"Error writing file: {exc}"

    @tool
    def list_directory(dir_path: str = ".") -> str:
        """List the contents of a directory.

        Returns files and subdirectories with their types.

        Args:
            dir_path: Path to the directory to list. Default is the workspace root.
        """
        target = Path(dir_path)
        if not target.is_absolute():
            target = workspace / target

        try:
            target = target.resolve()
            workspace_resolved = workspace.resolve()
            if not str(target).startswith(str(workspace_resolved)):
                return f"Error: Path {target} is outside the workspace."
        except Exception as exc:
            return f"Error resolving path: {exc}"

        if not target.exists():
            return f"Error: Directory does not exist: {target}"

        if not target.is_dir():
            return f"Error: Path is not a directory: {target}"

        try:
            entries: list[str] = []
            for entry in sorted(target.iterdir()):
                if entry.is_dir():
                    entries.append(f"  [DIR]  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    entries.append(f"  [FILE] {entry.name} ({size} bytes)")

            if not entries:
                return f"Directory {target} is empty."

            result = f"Contents of {target}:\n" + "\n".join(entries)
            logger.info("Listed directory: %s (%d entries)", target, len(entries))
            return result

        except Exception as exc:
            logger.error("Failed to list directory %s: %s", target, exc)
            return f"Error listing directory: {exc}"

    return [create_file, write_file, list_directory]
