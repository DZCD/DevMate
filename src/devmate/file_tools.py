"""File operation tools for DevMate.

Provides LangChain @tool-decorated functions for file system operations,
code search, and web fetching — ported from the TypeScript reference
implementations in agent-template-ts/src/tools/tools/.

Tools:
    read          — Read file contents with line numbers, binary/image detection
    write         — Create or overwrite files with diff output
    edit          — Fuzzy string replacement with 8-strategy chain
    glob          — Filename pattern matching sorted by mtime
    grep          — Regex content search across files
    bash          — Execute shell commands with timeout
    codesearch    — Exa API code search via MCP protocol
    websearch     — Exa AI web search via MCP protocol (direct)
    webfetch      — HTTP page fetching with HTML-to-text extraction
    create_file   — (deprecated) Create a new file
    list_directory — (deprecated) List directory contents
"""

import difflib
import fnmatch
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Generator

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_READ_BYTES = 50 * 1024  # 50 KB

BINARY_EXTENSIONS = frozenset(
    {
        ".zip",
        ".tar",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".class",
        ".jar",
        ".war",
        ".7z",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".bin",
        ".dat",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".wasm",
        ".pyc",
        ".pyo",
    }
)

IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".ico",
        ".tiff",
        ".heic",
        ".heif",
    }
)

MAX_BASH_OUTPUT = 10 * 1024  # 10 KB


# ---------------------------------------------------------------------------
# Helper: binary file detection
# ---------------------------------------------------------------------------


def _is_binary_file(file_path: str) -> bool:
    """Detect whether a file is binary by extension or content sampling."""
    ext = Path(file_path).suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return True
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False
    if not head:
        return False
    if b"\x00" in head:
        return True
    non_printable = sum(1 for b in head if b < 9 or (13 < b < 32))
    return non_printable / len(head) > 0.3


# ---------------------------------------------------------------------------
# Helper: HTML → plain text
# ---------------------------------------------------------------------------

_BLOCK_CLOSE_RE = re.compile(
    r"</(p|div|tr|li|h[1-6]|blockquote|section|article)>",
    re.IGNORECASE,
)
_BR_RE = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _extract_text_from_html(html: str) -> str:
    """Strip script/style tags, decode entities, convert block tags to newlines."""
    text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--[\s\S]*?-->", "", text, flags=re.DOTALL)
    text = _BLOCK_CLOSE_RE.sub("\n", text)
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Helper: str.replace with regex for multi-replace fallback
# ---------------------------------------------------------------------------


def _str_replace_all(text: str, old: str, new: str) -> str:
    """Replace *all* non-overlapping occurrences of *old* with *new*."""
    # Use re.escape so special chars in old are treated literally.
    return re.sub(re.escape(old), lambda _m: new, text)


# ===========================================================================
# Helper: glob-to-regex converter (for include filter)
# ===========================================================================


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern (with * ? {,}) to a compiled regex."""
    regex = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*" and i + 1 < len(pattern) and pattern[i + 1] == "*":
            if i + 2 < len(pattern) and pattern[i + 2] == "/":
                regex += "(?:.+/)?"
                i += 3
            else:
                regex += ".*"
                i += 2
        elif c == "*":
            regex += "[^/]*"
            i += 1
        elif c == "?":
            regex += "[^/]"
            i += 1
        elif c == "{":
            end = pattern.find("}", i)
            if end != -1:
                alternatives = pattern[i + 1 : end].split(",")
                regex += "(?:"
                regex += "|".join(alternatives)
                regex += ")"
                i = end + 1
            else:
                regex += re.escape(c)
                i += 1
        elif c in ".+^${}()|[]\\":
            regex += "\\" + c
            i += 1
        else:
            regex += c
            i += 1
    return re.compile(f"^{regex}$")


# ===========================================================================
# Tool definitions (defined inside factory so they capture *workspace*)
# ===========================================================================


def create_file_tools(workspace: str | Path | None = None) -> list[Any]:
    """Create and return all file operation tools.

    Args:
        workspace: Root directory for file operations.  Defaults to cwd.

    Returns:
        A list of LangChain ``@tool``-decorated callables.
    """
    if workspace is None:
        workspace = Path.cwd()
    workspace = Path(workspace)

    # ------------------------------------------------------------------
    # 1. read
    # ------------------------------------------------------------------
    @tool
    def read(file_path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT) -> str:
        """Reads a file from the local filesystem.

        Assume this tool is able to read all files on the machine. If the
        User provides a path to a file assume that path is valid. It is okay
        to read a file that does not exist; an error will be returned.

        Usage:
        - The filePath parameter must be an absolute path, not a relative path
        - By default, it reads up to 2000 lines starting from the beginning
          of the file
        - You can optionally specify a line offset and limit (especially handy
          for long files), but it's recommended to read the whole file by not
          providing these parameters
        - Any lines longer than 2000 characters will be truncated
        - Results are returned using cat -n format, with line numbers
          starting at 1
        - You have the capability to call multiple tools in a single response.
          It is always better to speculatively read multiple files as a batch
          that are potentially useful.
        - If you read a file that exists but has empty contents you will
          receive a system reminder warning in place of file contents.
        - This tool CANNOT read image files. For image files (.png, .jpg,
          .jpeg, .gif, .webp, etc.), use the image_understand tool instead.

        Args:
            file_path: Absolute path of the file to read.
            offset: 0-based starting line number.
            limit: Maximum number of lines to read.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()

        # File existence -------------------------------------------------
        if not target.exists():
            # Try to suggest similar file names
            try:
                parent = target.parent
                base = target.name.lower()
                if parent.is_dir():
                    suggestions = [
                        str(parent / e)
                        for e in parent.iterdir()
                        if base in e.name.lower() or e.name.lower() in base
                    ][:3]
                    if suggestions:
                        hint = "\n".join(suggestions)
                        msg = (
                            f"File not found: {target}\n\n"
                            f"Did you mean one of these?\n{hint}"
                        )
                        return msg
            except OSError:
                pass
            return f"File not found: {target}"

        if not target.is_file():
            return f"Not a file: {target}"

        # Empty file -----------------------------------------------------
        if target.stat().st_size == 0:
            return (
                f"<system-reminder>File exists but has empty contents: "
                f"{target}</system-reminder>"
            )

        # Image file -----------------------------------------------------
        ext = target.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return (
                f"Cannot read image file with this tool. "
                f"Use the image_understand tool instead to analyze image: {target}"
            )

        # Binary file ----------------------------------------------------
        if _is_binary_file(str(target)):
            return f"Cannot read binary file: {target}"

        # Read text ------------------------------------------------------
        try:
            text = target.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read %s: %s", target, exc)
            return f"Failed to read file: {target}"

        all_lines = text.split("\n")
        raw_lines: list[str] = []
        total_bytes = 0
        truncated_by_bytes = False

        for i in range(offset, min(len(all_lines), offset + limit)):
            line = all_lines[i]
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."
            line_size = len(line.encode("utf-8")) + (1 if raw_lines else 0)
            if total_bytes + line_size > MAX_READ_BYTES:
                truncated_by_bytes = True
                break
            raw_lines.append(line)
            total_bytes += line_size

        numbered = [
            f"{idx + offset + 1:05d}| {line}" for idx, line in enumerate(raw_lines)
        ]

        output_parts = ["<file>", "\n".join(numbered)]

        last_read_line = offset + len(raw_lines)
        has_more = len(all_lines) > last_read_line

        if truncated_by_bytes:
            output_parts.append(
                f"\n\n(Output truncated at {MAX_READ_BYTES} bytes. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        elif has_more:
            output_parts.append(
                f"\n\n(File has more lines. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        else:
            output_parts.append(f"\n\n(End of file - total {len(all_lines)} lines)")
        output_parts.append("</file>")

        return "".join(output_parts)

    # ------------------------------------------------------------------
    # 2. write
    # ------------------------------------------------------------------
    @tool
    def write(file_path: str, content: str) -> str:
        """Writes a file to the local filesystem.

        Usage:
        - This tool will overwrite the existing file if there is one at the
          provided path.
        - If this is an existing file, you MUST use the Read tool first to
          read the file's contents.
        - ALWAYS prefer editing existing files in the codebase. NEVER write
          new files unless explicitly required.
        - NEVER proactively create documentation files (*.md) or README
          files. Only create documentation files if explicitly requested by
          the User.
        - Only use emojis if the user explicitly requests it. Avoid writing
          emojis to files unless asked.
        - The filePath parameter must be an absolute path, not a relative
          path.
        - Parent directories will be created automatically if they don't
          exist.

        Args:
            file_path: Absolute path of the file to write.
            content: Full content to write to the file.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()

        # Read old content for diff (if file exists)
        old_content = ""
        exists = target.is_file()
        if exists:
            try:
                old_content = target.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not read old content of %s: %s", target, exc)

        # Write file
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to write %s: %s", target, exc)
            return f"Error writing file: {exc}"

        # Build result with optional diff
        result = f"File written successfully: {target}"
        if exists:
            diff_lines = list(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=str(target),
                    tofile=str(target),
                )
            )
            if diff_lines:
                diff_text = "".join(diff_lines)
                if diff_text.count("\n") > 50:
                    truncated = "\n".join(diff_text.split("\n")[:50])
                    remainder = diff_text.count("\n") - 50
                    suffix = f"\n... (diff truncated, {remainder} more lines)"
                    diff_text = truncated + suffix
                result += f" (overwritten)\n\n<diff>\n{diff_text}\n</diff>"
            else:
                result += " (no changes)"
        else:
            result += f" (new file, {len(content)} chars)"

        action = "new" if not exists else "overwritten"
        logger.info("Wrote file: %s (%s)", target, action)
        return result

    # ------------------------------------------------------------------
    # 3. edit — core fuzzy replacement with 8-strategy chain
    # ------------------------------------------------------------------

    # -- Levenshtein distance ------------------------------------------
    def _levenshtein(a: str, b: str) -> int:
        if not a or not b:
            return max(len(a), len(b))
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
            prev = curr
        return prev[-1]

    # -- Replacer type -------------------------------------------------
    ReplacerFunc = Generator[str, None, None]

    def _simple_replacer(_content: str, find: str) -> ReplacerFunc:
        yield find

    def _line_trimmed_replacer(content: str, find: str) -> ReplacerFunc:
        original_lines = content.split("\n")
        search_lines = find.split("\n")
        if search_lines and search_lines[-1] == "":
            search_lines.pop()

        for i in range(len(original_lines) - len(search_lines) + 1):
            match = True
            for j in range(len(search_lines)):
                if original_lines[i + j].strip() != search_lines[j].strip():
                    match = False
                    break
            if match:
                # Compute start/end character offsets
                start_idx = sum(len(original_lines[k]) + 1 for k in range(i))
                end_idx = start_idx
                for k in range(len(search_lines)):
                    end_idx += len(original_lines[i + k])
                    if k < len(search_lines) - 1:
                        end_idx += 1
                yield content[start_idx:end_idx]

    def _block_anchor_replacer(content: str, find: str) -> ReplacerFunc:
        original_lines = content.split("\n")
        search_lines = find.split("\n")
        if len(search_lines) < 3:
            return
        if search_lines and search_lines[-1] == "":
            search_lines.pop()

        first_trimmed = search_lines[0].strip()
        last_trimmed = search_lines[-1].strip()
        search_block_size = len(search_lines)

        # Collect candidates (first-line & last-line anchor pairs)
        candidates: list[tuple[int, int]] = []
        for i in range(len(original_lines)):
            if original_lines[i].strip() != first_trimmed:
                continue
            for j in range(i + 2, len(original_lines)):
                if original_lines[j].strip() == last_trimmed:
                    candidates.append((i, j))
                    break

        if not candidates:
            return

        def _similarity(start_line: int, end_line: int) -> float:
            actual_block_size = end_line - start_line + 1
            lines_to_check = min(search_block_size - 2, actual_block_size - 2)
            if lines_to_check <= 0:
                return 1.0
            sim = 0.0
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                ol = original_lines[start_line + j].strip()
                sl = search_lines[j].strip()
                max_len = max(len(ol), len(sl))
                if max_len == 0:
                    continue
                dist = _levenshtein(ol, sl)
                sim += 1 - dist / max_len
            return sim / lines_to_check

        def _block_text(start_line: int, end_line: int) -> str:
            s = sum(len(original_lines[k]) + 1 for k in range(start_line))
            e = s
            for k in range(start_line, end_line + 1):
                e += len(original_lines[k])
                if k < end_line:
                    e += 1
            return content[s:e]

        if len(candidates) == 1:
            start, end = candidates[0]
            sim = _similarity(start, end)
            if sim >= 0.0:
                yield _block_text(start, end)
            return

        # Multiple candidates — pick best similarity
        best: tuple[int, int] | None = None
        max_sim = -1.0
        for start, end in candidates:
            sim = _similarity(start, end)
            if sim > max_sim:
                max_sim = sim
                best = (start, end)
        if max_sim >= 0.3 and best is not None:
            yield _block_text(best[0], best[1])

    def _whitespace_normalized_replacer(content: str, find: str) -> ReplacerFunc:
        def _norm(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip()

        norm_find = _norm(find)
        lines = content.split("\n")

        for line in lines:
            if _norm(line) == norm_find:
                yield line
            else:
                norm_line = _norm(line)
                if norm_find and norm_find in norm_line:
                    words = find.strip().split()
                    if words:
                        pattern = r"\s+".join(re.escape(w) for w in words)
                        try:
                            m = re.search(pattern, line)
                            if m:
                                yield m.group(0)
                        except re.error:
                            pass

        find_lines = find.split("\n")
        if len(find_lines) > 1:
            for i in range(len(lines) - len(find_lines) + 1):
                block = "\n".join(lines[i : i + len(find_lines)])
                if _norm(block) == norm_find:
                    yield block

    def _indentation_flexible_replacer(content: str, find: str) -> ReplacerFunc:
        def _remove_indent(text: str) -> str:
            lines = text.split("\n")
            non_empty = [ln for ln in lines if ln.strip()]
            if not non_empty:
                return text
            min_ind = min(
                len(re.match(r"^(\s*)", ln).group(1))  # type: ignore[arg-type]
                for ln in non_empty
            )
            return "\n".join(ln if not ln.strip() else ln[min_ind:] for ln in lines)

        norm_find = _remove_indent(find)
        content_lines = content.split("\n")
        find_lines = find.split("\n")

        for i in range(len(content_lines) - len(find_lines) + 1):
            block = "\n".join(content_lines[i : i + len(find_lines)])
            if _remove_indent(block) == norm_find:
                yield block

    def _escape_normalized_replacer(content: str, find: str) -> ReplacerFunc:
        def _unescape(s: str) -> str:
            return re.sub(
                r"\\(n|t|r|'|\"|`|\\|\n|\$)",
                lambda m: {
                    "n": "\n",
                    "t": "\t",
                    "r": "\r",
                    "'": "'",
                    '"': '"',
                    "`": "`",
                    "\\": "\\",
                    "\n": "\n",
                    "$": "$",
                }.get(m.group(1), m.group(0)),
                s,
            )

        unesc_find = _unescape(find)
        if unesc_find in content:
            yield unesc_find

        lines = content.split("\n")
        find_lines = unesc_find.split("\n")
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if _unescape(block) == unesc_find:
                yield block

    def _trimmed_boundary_replacer(content: str, find: str) -> ReplacerFunc:
        trimmed = find.strip()
        if trimmed == find:
            return

        if trimmed in content:
            yield trimmed

        lines = content.split("\n")
        find_lines = find.split("\n")
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if block.strip() == trimmed:
                yield block

    def _context_aware_replacer(content: str, find: str) -> ReplacerFunc:
        find_lines = find.split("\n")
        if len(find_lines) < 3:
            return
        if find_lines and find_lines[-1] == "":
            find_lines.pop()

        content_lines = content.split("\n")
        first_line = find_lines[0].strip()
        last_line = find_lines[-1].strip()

        for i in range(len(content_lines)):
            if content_lines[i].strip() != first_line:
                continue
            for j in range(i + 2, len(content_lines)):
                if content_lines[j].strip() == last_line:
                    block_lines = content_lines[i : j + 1]
                    if len(block_lines) == len(find_lines):
                        matching = 0
                        total_non_empty = 0
                        for k in range(1, len(block_lines) - 1):
                            bl = block_lines[k].strip()
                            fl = find_lines[k].strip()
                            if bl or fl:
                                total_non_empty += 1
                                if bl == fl:
                                    matching += 1
                        if total_non_empty == 0 or matching / total_non_empty >= 0.5:
                            yield "\n".join(block_lines)
                    break

    # -- Ordered replacer chain (9 strategies) -------------------------
    _REPLACERS: list = [
        _simple_replacer,
        _line_trimmed_replacer,
        _block_anchor_replacer,
        _whitespace_normalized_replacer,
        _indentation_flexible_replacer,
        _escape_normalized_replacer,
        _trimmed_boundary_replacer,
        _context_aware_replacer,
        # MultiOccurrence is handled inline after the loop
    ]

    def _edit_replace(
        content: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Apply the fuzzy replacement strategy chain."""
        if old_string == new_string:
            raise ValueError("oldString and newString must be different")

        not_found = True

        for replacer in _REPLACERS:
            for search in replacer(content, old_string):
                idx = content.find(search)
                if idx == -1:
                    continue
                not_found = False
                if replace_all:
                    return _str_replace_all(content, search, new_string)
                last_idx = content.rfind(search)
                if idx == last_idx:
                    return content[:idx] + new_string + content[idx + len(search) :]
                # Multiple occurrences — ambiguity error handled after loop
                break  # move to next replacer

        # MultiOccurrence (exact string, find all positions)
        if not_found:
            start = 0
            positions: list[int] = []
            while True:
                pos = content.find(old_string, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + len(old_string)
            if positions:
                if replace_all:
                    return _str_replace_all(content, old_string, new_string)
                not_found = False

        if not_found:
            raise ValueError("oldString not found in content")

        raise ValueError(
            "Found multiple matches for oldString. Provide more surrounding "
            "lines in oldString to identify the correct match."
        )

    def _trim_diff(diff_text: str) -> str:
        """Remove common leading indentation from diff hunks."""
        lines = diff_text.split("\n")
        content_lines = [
            ln
            for ln in lines
            if (ln.startswith("+") or ln.startswith("-") or ln.startswith(" "))
            and not ln.startswith("---")
            and not ln.startswith("+++")
        ]
        if not content_lines:
            return diff_text

        min_indent = float("inf")
        for line in content_lines:
            body = line[1:]
            if body.strip():
                m = re.match(r"^(\s*)", body)
                if m:
                    min_indent = min(min_indent, len(m.group(1)))
        if min_indent == float("inf") or min_indent == 0:
            return diff_text

        result: list[str] = []
        for line in lines:
            if (
                (line.startswith("+") or line.startswith("-") or line.startswith(" "))
                and not line.startswith("---")
                and not line.startswith("+++")
            ):
                result.append(line[0] + line[1:][min_indent:])
            else:
                result.append(line)
        return "\n".join(result)

    @tool
    def edit(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Performs exact string replacements in files.

        Usage:
        - You MUST use the Read tool first to read the file's contents before
          editing.
        - The edit will FAIL if oldString is not found in the file. Provide
          enough surrounding context to make it unique.
        - oldString and newString must be different.
        - Use replaceAll to replace every occurrence of oldString (default:
          only replace a single unique match).
        - When oldString is empty and the file does not exist, a new file
          will be created with newString as content.
        - The filePath parameter must be an absolute path, not a relative
          path.

        Args:
            file_path: Absolute path of the file to edit.
            old_string: The text to find (must uniquely match).
            new_string: The replacement text (must differ from old_string).
            replace_all: If True, replace every occurrence.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()

        if old_string == new_string:
            return "Error: oldString and newString must be different"

        # Empty old_string → create new file
        if not old_string:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(new_string, encoding="utf-8")
                logger.info("Created new file via edit: %s", target)
                return f"New file created: {target} ({len(new_string)} chars)"
            except Exception as exc:
                logger.error("Failed to create file %s: %s", target, exc)
                return f"Error creating file: {exc}"

        # Read existing file
        if not target.is_file():
            return f"Error: File does not exist: {target}"
        try:
            old_content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading file: {exc}"

        # Execute replacement
        try:
            new_content = _edit_replace(
                old_content,
                old_string,
                new_string,
                replace_all,
            )
        except ValueError as exc:
            return f"Edit failed: {exc}"

        # Write back
        try:
            target.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to write %s: %s", target, exc)
            return f"Error writing file: {exc}"

        # Generate diff
        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=str(target),
                tofile=str(target),
            )
        )
        if diff_lines:
            diff_text = _trim_diff("".join(diff_lines))
            diff_array = diff_text.split("\n")
            if len(diff_array) > 80:
                truncated = "\n".join(diff_array[:80])
                remainder = len(diff_array) - 80
                suffix = f"\n... (diff truncated, {remainder} more lines)"
                diff_text = truncated + suffix
        else:
            diff_text = ""

        logger.info("Edited file: %s", target)
        result = f"Edit successful: {target}"
        if diff_text:
            result += f"\n\n<diff>\n{diff_text}\n</diff>"
        return result

    # ------------------------------------------------------------------
    # 4. glob
    # ------------------------------------------------------------------
    @tool
    def glob(pattern: str, path: str | None = None) -> str:
        """Fast file pattern matching tool supporting glob patterns.

        Supports glob patterns like "**/*.ts", "src/**/*.js".
        Returns matching file paths sorted by modification time (newest first).
        Useful for finding files by name pattern.

        Args:
            pattern: Glob pattern to match, e.g. **/*.ts
            path: Search directory (optional, defaults to workspace).
        """
        search_dir = Path(path) if path else workspace
        if not search_dir.is_absolute():
            search_dir = workspace / search_dir
        search_dir = search_dir.resolve()

        LIMIT = 100
        files: list[tuple[str, float]] = []
        truncated = False

        try:
            for entry in search_dir.rglob("*"):
                normalized = entry.as_posix()
                if not fnmatch.fnmatch(normalized, pattern):
                    continue
                if not entry.is_file():
                    continue
                try:
                    mtime = entry.stat().st_mtime
                    files.append((str(entry), mtime))
                    if len(files) >= LIMIT:
                        truncated = True
                        break
                except OSError:
                    continue
        except OSError:
            return f"Directory does not exist or cannot be accessed: {search_dir}"

        files.sort(key=lambda x: x[1], reverse=True)

        if not files:
            return "No matching files found"

        output = [f[0] for f in files]
        if truncated:
            output.append("")
            output.append("(Results truncated, use a more specific pattern)")
        return "\n".join(output)

    # ------------------------------------------------------------------
    # 5. grep
    # ------------------------------------------------------------------
    @tool
    def grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
        """Fast content search tool that works with any codebase size.

        Searches file contents using regular expressions.
        Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.).
        Filter files by pattern with the include parameter (eg. "*.js",
        "*.{ts,tsx}").
        Returns file paths and line numbers with at least one match sorted by
        modification time.
        Use this tool when you need to find files containing specific patterns.
        If you need to identify/count the number of matches within files, use
        the Bash tool with `rg` (ripgrep) directly. Do NOT use `grep`.

        Args:
            pattern: Regular expression to search for.
            path: Search directory (optional, defaults to workspace).
            include: File filter pattern (optional), e.g. "*.js", "*.{ts,tsx}".
        """
        if not pattern:
            return "pattern parameter cannot be empty"

        search_dir = Path(path) if path else workspace
        if not search_dir.is_absolute():
            search_dir = workspace / search_dir
        search_dir = search_dir.resolve()

        try:
            regex = re.compile(pattern)
        except re.error:
            return f"Invalid regex: {pattern}"

        include_regex = _glob_to_regex(include) if include else None
        LIMIT = 100

        matches: list[tuple[str, float, int, str]] = []
        truncated = False

        try:
            for entry in search_dir.rglob("*"):
                if not entry.is_file():
                    continue
                try:
                    if include_regex:
                        if not include_regex.match(entry.name):
                            continue
                    stat_result = entry.stat()
                    content = entry.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if regex.search(line):
                        if len(matches) >= LIMIT:
                            truncated = True
                            break
                        matches.append(
                            (
                                str(entry),
                                stat_result.st_mtime,
                                i + 1,
                                line,
                            )
                        )
                if truncated:
                    break
        except OSError:
            return f"Directory does not exist or cannot be accessed: {search_dir}"

        if not matches:
            return "No matches found"

        matches.sort(key=lambda x: x[1], reverse=True)

        output_parts = [f"Found {len(matches)} matches"]
        current_file = ""
        for match_path, _mtime, line_num, line_text in matches:
            if match_path != current_file:
                if current_file:
                    output_parts.append("")
                current_file = match_path
                output_parts.append(f"{match_path}:")
            trunc_mark = "..." if len(line_text) > MAX_LINE_LENGTH else ""
            text = line_text[:MAX_LINE_LENGTH] + trunc_mark if trunc_mark else line_text
            output_parts.append(f"  Line {line_num}: {text}")

        if truncated:
            output_parts.append("")
            output_parts.append("(Results truncated, use a more specific pattern)")

        return "\n".join(output_parts)

    # ------------------------------------------------------------------
    # 6. bash
    # ------------------------------------------------------------------
    @tool
    def bash(command: str, cwd: str | None = None, timeout: int = 30000) -> str:
        """Execute a shell command.

        Use for running shell commands (ls, cat, mkdir, npm, git, etc.),
        executing scripts, build tasks, checking system environment, etc.

        Args:
            command: The shell command to execute.
            cwd: Working directory (optional, defaults to workspace).
            timeout: Timeout in milliseconds (default 30000).
        """
        if not command or not command.strip():
            return "[bash] Error: command cannot be empty"

        effective_cwd = cwd if cwd else str(workspace)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=effective_cwd,
                timeout=timeout / 1000,
                env={**os.environ, "PAGER": "cat"},
            )
        except subprocess.TimeoutExpired:
            return f"[bash] Command timed out ({timeout}ms), terminated"
        except Exception as exc:
            return f"[bash] Error: {exc}"

        output_parts: list[str] = []

        if result.stdout:
            stdout = result.stdout
            if len(stdout) > MAX_BASH_OUTPUT:
                trunc = f"\n... (output truncated, {len(result.stdout)} chars total)"
                stdout = stdout[:MAX_BASH_OUTPUT] + trunc
            output_parts.append(stdout)

        if result.stderr:
            stderr = result.stderr
            if len(stderr) > MAX_BASH_OUTPUT:
                trunc = f"\n... (stderr truncated, {len(result.stderr)} chars total)"
                stderr = stderr[:MAX_BASH_OUTPUT] + trunc
            output_parts.append(f"[stderr]\n{stderr}")

        if result.returncode != 0 and not result.stderr:
            output_parts.append(f"[bash] Exit code: {result.returncode}")

        if not output_parts:
            output_parts.append("[bash] Command executed successfully (no output)")

        return "\n".join(output_parts)

    # ------------------------------------------------------------------
    # 7. codesearch
    # ------------------------------------------------------------------
    @tool
    def codesearch(query: str, tokens_num: int = 5000) -> str:
        """Search and get relevant context for any programming task using Exa Code API.

        Provides high-quality, fresh context for libraries, SDKs, and APIs.
        Returns comprehensive code examples, documentation, and API references.

        Usage notes:
          - Adjustable token count (1000-50000) for focused or comprehensive
            results
          - Default 5000 tokens provides balanced context for most queries
          - Use lower values for specific questions, higher values for
            comprehensive documentation

        Args:
            query: Search query for finding API, library, and SDK context.
                Examples: 'React useState hook examples',
                'Python pandas dataframe filtering'
            tokens_num: Number of tokens to return (1000-50000), default 5000.
        """
        tokens_num = max(1000, min(50000, tokens_num))

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_code_context_exa",
                "arguments": {"query": query, "tokensNum": tokens_num},
            },
        }

        try:
            import httpx
        except ImportError:
            return (
                "codesearch requires the 'httpx' package. "
                "Install with: pip install httpx"
            )

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://mcp.exa.ai/mcp",
                    json=payload,
                    headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                    },
                )
            if response.status_code != 200:
                err = response.text[:500]
                return f"Code search request failed ({response.status_code}): {err}"

            # Parse SSE response
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        content = data.get("result", {}).get("content", [])
                        if content and content[0].get("text"):
                            return content[0]["text"]
                    except (ValueError, KeyError, IndexError):
                        continue

            return "No relevant code or documentation found. Try a more specific query."
        except httpx.TimeoutException:
            return "Code search request timed out (30s)"
        except Exception as exc:
            logger.error("codesearch failed: %s", exc)
            return f"Code search failed: {exc}"

    # ------------------------------------------------------------------
    # 8. websearch — Exa AI web search via MCP protocol (direct)
    # ------------------------------------------------------------------
    @tool
    def websearch(
        query: str,
        num_results: int = 8,
        livecrawl: str = "fallback",
        type: str = "auto",
        context_max_characters: int = 10000,
    ) -> str:
        """Web search tool powered by Exa AI.

        Search the internet for up-to-date information, documentation,
        articles, and more. Today's date is 2026-04-07. Returns search
        results with relevant content excerpts.

        Use this when you need current information that may not be in
        your training data.

        Supports different search modes:
          - 'auto' (balanced, default)
          - 'fast' (quick results)
          - 'deep' (comprehensive)

        Args:
            query: Search query keywords.
            num_results: Number of results to return (default 8).
            livecrawl: Crawl mode: 'fallback' (default) or 'preferred'.
            type: Search type: 'auto', 'fast', or 'deep'.
            context_max_characters: Max context chars per result (default 10000).
        """
        num_results = max(1, min(20, num_results))
        if livecrawl not in ("fallback", "preferred"):
            livecrawl = "fallback"
        if type not in ("auto", "fast", "deep"):
            type = "auto"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "type": type,
                    "numResults": num_results,
                    "livecrawl": livecrawl,
                    "contextMaxCharacters": context_max_characters,
                },
            },
        }

        try:
            import httpx
        except ImportError:
            return (
                "websearch requires the 'httpx' package. "
                "Install with: pip install httpx"
            )

        try:
            with httpx.Client(timeout=25.0) as client:
                response = client.post(
                    "https://mcp.exa.ai/mcp",
                    json=payload,
                    headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                    },
                )
            if response.status_code != 200:
                err = response.text[:500]
                return f"Web search request failed ({response.status_code}): {err}"

            # Parse SSE response
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        content = data.get("result", {}).get("content", [])
                        if content and content[0].get("text"):
                            return content[0]["text"]
                    except (ValueError, KeyError, IndexError):
                        continue

            return "No search results found. Try a different query."
        except httpx.TimeoutException:
            return "Web search request timed out (25s)"
        except Exception as exc:
            logger.error("websearch failed: %s", exc)
            return f"Web search failed: {exc}"

    # ------------------------------------------------------------------
    # 10. webfetch
    # ------------------------------------------------------------------
    @tool
    def webfetch(url: str, max_chars: int = 50000) -> str:
        """Fetches the content of a web page given its URL.

        Usage:
        - Use this tool to retrieve the text content of a specific web page
          when you already know the URL.
        - For general information discovery, prefer the 'websearch' tool
          instead.
        - Supports HTML pages — the tool will attempt to extract readable
          text content.
        - You can optionally limit the maximum number of characters returned.
        - If the page requires JavaScript rendering, the result may be
          incomplete.
        - Respects a 30-second timeout to avoid hanging on slow pages.
        - Do NOT use this tool for images.

        Args:
            url: Full URL to fetch (must start with http:// or https://).
            max_chars: Maximum characters to return (default 50000).
        """
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return (
                "Invalid URL format. Please provide a full address "
                "starting with http:// or https://."
            )

        try:
            import httpx
        except ImportError:
            return (
                "webfetch requires the 'httpx' package. Install with: pip install httpx"
            )

        logger.info("webfetch: fetching %s", url)

        try:
            with httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AgentFetch/1.0)",
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,*/*;q=0.8"
                    ),
                },
            ) as client:
                response = client.get(url)

            if response.status_code != 200:
                error_text = response.text[:500]
                return (
                    f"Web page request failed (HTTP "
                    f"{response.status_code}): {error_text}"
                )

            content_type = response.headers.get("content-type", "")
            raw_text = response.text

            if "text/html" in content_type or "application/xhtml" in content_type:
                content = _extract_text_from_html(raw_text)
            else:
                content = raw_text

            truncated = len(content) > max_chars
            result = content[:max_chars] if truncated else content
            suffix = (
                f"\n\n... (content truncated, original {len(content)} chars, "
                f"returned first {max_chars} chars)"
                if truncated
                else ""
            )

            logger.info("webfetch: fetched %s, content length: %d", url, len(content))
            return f'<webfetch url="{url}">\n{result}{suffix}\n</webfetch>'

        except httpx.TimeoutException:
            logger.warning("webfetch: request timed out (30s) for %s", url)
            return (
                "Web page request timed out (30s). "
                "Please check if the URL is accessible."
            )
        except Exception as exc:
            logger.error("webfetch failed: %s", exc)
            return f"Web fetch failed: {exc}"

    # ------------------------------------------------------------------
    # 11. create_file (deprecated, retained for backward compatibility)
    # ------------------------------------------------------------------
    @tool
    def create_file(file_path: str, content: str = "", overwrite: bool = False) -> str:
        """[DEPRECATED] Use the 'write' tool instead.

        Create a new file at the specified path.

        Args:
            file_path: Relative or absolute path for the new file.
            content: Initial content to write to the file.
            overwrite: If True, overwrite existing files. Default False.
        """
        target = Path(file_path)
        if not target.is_absolute():
            target = workspace / target

        try:
            target = target.resolve()
        except Exception as exc:
            return f"Error resolving path: {exc}"

        if target.exists() and not overwrite:
            return (
                f"Error: File already exists: {target}. Use overwrite=True to replace."
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            logger.info("Created file (deprecated tool): %s", target)
            return f"Successfully created file: {target}"
        except Exception as exc:
            logger.error("Failed to create file %s: %s", target, exc)
            return f"Error creating file: {exc}"

    # ------------------------------------------------------------------
    # 12. list_directory (deprecated, retained for backward compatibility)
    # ------------------------------------------------------------------
    @tool
    def list_directory(dir_path: str = ".") -> str:
        """[DEPRECATED] Use the 'bash' tool with 'ls' instead.

        List the contents of a directory.

        Args:
            dir_path: Path to the directory to list. Default is the workspace.
        """
        target = Path(dir_path)
        if not target.is_absolute():
            target = workspace / target

        try:
            target = target.resolve()
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

    # ------------------------------------------------------------------
    # Return all tools
    # ------------------------------------------------------------------
    return [
        read,
        write,
        edit,
        glob,
        grep,
        bash,
        codesearch,
        websearch,
        webfetch,
        create_file,
        list_directory,
    ]
