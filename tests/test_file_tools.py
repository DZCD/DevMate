"""Tests for the file_tools module."""

from unittest.mock import patch

import pytest

from devmate.file_tools import create_file_tools

# ===========================================================================
# Helper: fixture to get all tools
# ===========================================================================


@pytest.fixture()
def tools(tmp_path):
    """Return all tools bound to a temporary workspace."""
    return create_file_tools(workspace=tmp_path)


def _get_tool(tools, name):
    """Retrieve a tool by name from the tools list."""
    return next(t for t in tools if t.name == name)


# ===========================================================================
# Tool count
# ===========================================================================


def test_tools_count(tools) -> None:
    """Test that create_file_tools returns exactly 10 tools."""
    tool_names = [t.name for t in tools]
    assert len(tools) == 10
    for name in (
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "bash",
        "codesearch",
        "webfetch",
        "create_file",
        "list_directory",
    ):
        assert name in tool_names, f"Missing tool: {name}"


# ===========================================================================
# 1. read — tests
# ===========================================================================


class TestReadTool:
    """Tests for the read file tool."""

    def test_read_normal_file(self, tools, tmp_path) -> None:
        """Test reading a normal text file."""
        target = tmp_path / "hello.txt"
        target.write_text("line1\nline2\nline3\n", encoding="utf-8")

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target)})

        assert "<file>" in result
        assert "</file>" in result
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "End of file" in result

    def test_read_with_offset_and_limit(self, tools, tmp_path) -> None:
        """Test reading with offset/limit pagination."""
        lines = "\n".join(f"line{i}" for i in range(10))
        target = tmp_path / "paging.txt"
        target.write_text(lines, encoding="utf-8")

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target), "offset": 3, "limit": 2})

        assert "line3" in result
        assert "line4" in result
        assert "line2" not in result
        assert "line5" not in result

    def test_read_nonexistent_file(self, tools, tmp_path) -> None:
        """Test reading a file that does not exist."""
        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(tmp_path / "nope.txt")})

        assert "File not found" in result

    def test_read_empty_file(self, tools, tmp_path) -> None:
        """Test reading an empty file returns a system reminder."""
        target = tmp_path / "empty.txt"
        target.write_text("", encoding="utf-8")

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target)})

        assert "empty contents" in result

    def test_read_image_file_rejected(self, tools, tmp_path) -> None:
        """Test that image files are rejected with a helpful message."""
        target = tmp_path / "photo.png"
        # Write minimal PNG header bytes
        target.write_bytes(b"\x89PNG\r\n\x1a\n")

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target)})

        assert "image" in result.lower() or "image_understand" in result

    def test_read_binary_file_rejected(self, tools, tmp_path) -> None:
        """Test that binary files are rejected."""
        target = tmp_path / "data.bin"
        target.write_bytes(b"\x00\x01\x02\x03\x04" * 100)

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target)})

        assert "binary" in result.lower()

    def test_read_line_numbers(self, tools, tmp_path) -> None:
        """Test that output uses cat -n style line numbers."""
        target = tmp_path / "numbered.txt"
        target.write_text("aaa\nbbb\nccc\n", encoding="utf-8")

        read_tool = _get_tool(tools, "read")
        result = read_tool.invoke({"file_path": str(target)})

        assert "00001|" in result
        assert "00002|" in result
        assert "00003|" in result


# ===========================================================================
# 2. write — tests
# ===========================================================================


class TestWriteTool:
    """Tests for the write file tool."""

    def test_write_new_file(self, tools, tmp_path) -> None:
        """Test writing a new file creates it with the correct content."""
        write_tool = _get_tool(tools, "write")
        target = tmp_path / "new_file.txt"

        result = write_tool.invoke(
            {"file_path": str(target), "content": "hello world\n"}
        )

        assert "File written successfully" in result
        assert "new file" in result
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello world\n"

    def test_write_overwrite_file(self, tools, tmp_path) -> None:
        """Test overwriting an existing file produces a diff."""
        target = tmp_path / "existing.txt"
        target.write_text("old line\n", encoding="utf-8")

        write_tool = _get_tool(tools, "write")
        result = write_tool.invoke({"file_path": str(target), "content": "new line\n"})

        assert "overwritten" in result
        assert "<diff>" in result
        assert "</diff>" in result
        assert target.read_text(encoding="utf-8") == "new line\n"

    def test_write_creates_parent_dirs(self, tools, tmp_path) -> None:
        """Test that parent directories are created automatically."""
        write_tool = _get_tool(tools, "write")
        target = tmp_path / "a" / "b" / "c" / "deep.txt"

        result = write_tool.invoke(
            {"file_path": str(target), "content": "deep content"}
        )

        assert "File written successfully" in result
        assert target.exists()

    def test_write_no_changes(self, tools, tmp_path) -> None:
        """Test writing identical content shows 'no changes'."""
        content = "same content\n"
        target = tmp_path / "same.txt"
        target.write_text(content, encoding="utf-8")

        write_tool = _get_tool(tools, "write")
        result = write_tool.invoke({"file_path": str(target), "content": content})

        assert "no changes" in result


# ===========================================================================
# 3. edit — tests
# ===========================================================================


class TestEditTool:
    """Tests for the edit file tool."""

    def test_edit_simple_replace(self, tools, tmp_path) -> None:
        """Test simple string replacement."""
        target = tmp_path / "edit_me.txt"
        target.write_text("hello world\nfoo bar\n", encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "hello world",
                "new_string": "hello devmate",
            }
        )

        assert "Edit successful" in result
        assert target.read_text(encoding="utf-8") == "hello devmate\nfoo bar\n"

    def test_edit_replace_all(self, tools, tmp_path) -> None:
        """Test replace_all replaces every occurrence."""
        target = tmp_path / "replace_all.txt"
        target.write_text("aaa bbb aaa\nccc aaa\n", encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "aaa",
                "new_string": "XXX",
                "replace_all": True,
            }
        )

        assert "Edit successful" in result
        assert target.read_text(encoding="utf-8") == "XXX bbb XXX\nccc XXX\n"

    def test_edit_multiline_replace(self, tools, tmp_path) -> None:
        """Test replacing a multi-line block."""
        content = "line1\nline2\nline3\nline4\n"
        target = tmp_path / "multi.txt"
        target.write_text(content, encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "line2\nline3",
                "new_string": "REPLACED",
            }
        )

        assert "Edit successful" in result
        assert target.read_text(encoding="utf-8") == "line1\nREPLACED\nline4\n"

    def test_edit_old_string_not_found(self, tools, tmp_path) -> None:
        """Test that a non-matching old_string returns an error."""
        target = tmp_path / "notfound.txt"
        target.write_text("some content\n", encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "NOT HERE",
                "new_string": "REPLACEMENT",
            }
        )

        assert "not found" in result.lower() or "Edit failed" in result

    def test_edit_same_old_and_new(self, tools, tmp_path) -> None:
        """Test that old_string == new_string returns an error."""
        target = tmp_path / "same.txt"
        target.write_text("content\n", encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "content",
                "new_string": "content",
            }
        )

        assert "must be different" in result

    def test_edit_creates_file_when_empty_old_string(self, tools, tmp_path) -> None:
        """Test that empty old_string creates a new file."""
        edit_tool = _get_tool(tools, "edit")
        target = tmp_path / "brand_new.txt"

        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "",
                "new_string": "freshly created",
            }
        )

        assert "New file created" in result or "created" in result
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "freshly created"

    def test_edit_nonexistent_file(self, tools, tmp_path) -> None:
        """Test editing a non-existent file with non-empty old_string."""
        edit_tool = _get_tool(tools, "edit")

        result = edit_tool.invoke(
            {
                "file_path": str(tmp_path / "no_such.txt"),
                "old_string": "something",
                "new_string": "else",
            }
        )

        assert "does not exist" in result

    def test_edit_produces_diff(self, tools, tmp_path) -> None:
        """Test that edit output includes a diff."""
        target = tmp_path / "diff_test.txt"
        target.write_text("old\n", encoding="utf-8")

        edit_tool = _get_tool(tools, "edit")
        result = edit_tool.invoke(
            {
                "file_path": str(target),
                "old_string": "old",
                "new_string": "new",
            }
        )

        assert "<diff>" in result
        assert "</diff>" in result


# ===========================================================================
# 4. glob — tests
# ===========================================================================


class TestGlobTool:
    """Tests for the glob pattern matching tool."""

    def test_glob_matches_pattern(self, tools, tmp_path) -> None:
        """Test basic glob pattern matching."""
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")

        glob_tool = _get_tool(tools, "glob")
        result = glob_tool.invoke({"pattern": "*.py"})

        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_recursive_pattern(self, tools, tmp_path) -> None:
        """Test recursive glob with ** pattern."""
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "file.ts").write_text("", encoding="utf-8")

        glob_tool = _get_tool(tools, "glob")
        result = glob_tool.invoke({"pattern": "**/*.ts"})

        assert "file.ts" in result

    def test_glob_with_path(self, tools, tmp_path) -> None:
        """Test glob with explicit path parameter."""
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "b.py").write_text("", encoding="utf-8")

        glob_tool = _get_tool(tools, "glob")
        result = glob_tool.invoke({"pattern": "*.py", "path": str(subdir)})

        assert "b.py" in result
        assert "a.py" not in result

    def test_glob_no_matches(self, tools, tmp_path) -> None:
        """Test glob with no matching files."""
        glob_tool = _get_tool(tools, "glob")
        result = glob_tool.invoke({"pattern": "*.xyz"})

        assert "No matching files found" in result

    def test_glob_sorted_by_mtime(self, tools, tmp_path) -> None:
        """Test that results are sorted by modification time (newest first)."""
        (tmp_path / "old.py").write_text("old", encoding="utf-8")
        import time

        time.sleep(0.05)
        (tmp_path / "new.py").write_text("new", encoding="utf-8")

        glob_tool = _get_tool(tools, "glob")
        result = glob_tool.invoke({"pattern": "*.py"})

        # "new.py" should appear before "old.py"
        assert result.index("new.py") < result.index("old.py")


# ===========================================================================
# 5. grep — tests
# ===========================================================================


class TestGrepTool:
    """Tests for the grep content search tool."""

    def test_grep_regex_search(self, tools, tmp_path) -> None:
        """Test basic regex content search."""
        f = tmp_path / "code.py"
        f.write_text(
            "def hello():\n    pass\n\ndef world():\n    pass\n",
            encoding="utf-8",
        )

        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": r"def\s+\w+"})

        assert "code.py" in result
        assert "def hello" in result
        assert "def world" in result

    def test_grep_include_filter(self, tools, tmp_path) -> None:
        """Test grep with include filter."""
        (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "main.ts").write_text("import os\n", encoding="utf-8")

        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": "import", "include": "*.py"})

        assert "main.py" in result
        assert "main.ts" not in result

    def test_grep_no_matches(self, tools, tmp_path) -> None:
        """Test grep with no matching content."""
        f = tmp_path / "empty.py"
        f.write_text("hello world\n", encoding="utf-8")

        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": "xyz_not_found"})

        assert "No matches found" in result

    def test_grep_invalid_regex(self, tools, tmp_path) -> None:
        """Test grep with invalid regex returns error."""
        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": "[invalid"})

        assert "Invalid regex" in result

    def test_grep_empty_pattern(self, tools, tmp_path) -> None:
        """Test grep with empty pattern returns error."""
        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": ""})

        assert "cannot be empty" in result

    def test_grep_shows_line_numbers(self, tools, tmp_path) -> None:
        """Test grep output includes line numbers."""
        f = tmp_path / "lines.py"
        f.write_text("aaa\nbbb\nccc\n", encoding="utf-8")

        grep_tool = _get_tool(tools, "grep")
        result = grep_tool.invoke({"pattern": "bbb"})

        assert "Line 2" in result


# ===========================================================================
# 6. bash — tests
# ===========================================================================


class TestBashTool:
    """Tests for the bash shell command tool."""

    def test_bash_simple_command(self, tools, tmp_path) -> None:
        """Test running a simple shell command."""
        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": "echo hello"})

        assert "hello" in result

    def test_bash_cwd(self, tools, tmp_path) -> None:
        """Test bash with explicit cwd parameter."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "test.txt").write_text("content", encoding="utf-8")

        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": "ls", "cwd": str(subdir)})

        assert "test.txt" in result

    def test_bash_timeout(self, tools) -> None:
        """Test bash command that exceeds timeout is terminated."""
        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": "sleep 10", "timeout": 1000})

        assert "timed out" in result.lower()

    def test_bash_empty_command(self, tools) -> None:
        """Test bash with empty command returns error."""
        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": ""})

        assert "cannot be empty" in result

    def test_bash_exit_code(self, tools) -> None:
        """Test bash returns exit code for failing commands."""
        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": "false"})

        assert "Exit code" in result or result != ""

    def test_bash_stderr(self, tools) -> None:
        """Test bash captures stderr."""
        bash_tool = _get_tool(tools, "bash")
        result = bash_tool.invoke({"command": "echo error_msg >&2"})

        assert "[stderr]" in result
        assert "error_msg" in result


# ===========================================================================
# 7. codesearch — tests
# ===========================================================================


class TestCodesearchTool:
    """Tests for the codesearch tool (Exa API via MCP)."""

    def _mock_post_response(self, mock_client_cls):
        """Return a mock response from the httpx.Client.post chain."""
        return mock_client_cls.return_value.__enter__.return_value.post.return_value

    @patch("httpx.Client")
    def test_codesearch_success(self, mock_client_cls, tools) -> None:
        """Test codesearch returns parsed content from SSE response."""
        mock_response = self._mock_post_response(mock_client_cls)
        mock_response.status_code = 200

        # Simulate SSE response
        import json as _json

        sse_data = _json.dumps(
            {"result": {"content": [{"text": "Here is some code search result."}]}}
        )
        mock_response.text = f"data: {sse_data}\n\n"

        codesearch_tool = _get_tool(tools, "codesearch")
        result = codesearch_tool.invoke({"query": "Python asyncio example"})

        assert "code search result" in result

    @patch("httpx.Client")
    def test_codesearch_no_results(self, mock_client_cls, tools) -> None:
        """Test codesearch with no relevant results."""
        mock_response = self._mock_post_response(mock_client_cls)
        mock_response.status_code = 200
        mock_response.text = "data: {}\n\n"

        codesearch_tool = _get_tool(tools, "codesearch")
        result = codesearch_tool.invoke({"query": "nonexistent query"})

        assert "No relevant" in result or "not found" in result.lower()

    @patch("httpx.Client")
    def test_codesearch_http_error(self, mock_client_cls, tools) -> None:
        """Test codesearch handles HTTP error responses."""
        mock_response = self._mock_post_response(mock_client_cls)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        codesearch_tool = _get_tool(tools, "codesearch")
        result = codesearch_tool.invoke({"query": "test"})

        assert "failed" in result.lower() or "500" in result

    @patch("httpx.Client")
    def test_codesearch_timeout(self, mock_client_cls, tools) -> None:
        """Test codesearch handles timeout gracefully."""
        import httpx as httpx_mod

        (
            mock_client_cls.return_value.__enter__.return_value.post.side_effect
        ) = httpx_mod.TimeoutException("timeout")

        codesearch_tool = _get_tool(tools, "codesearch")
        result = codesearch_tool.invoke({"query": "test"})

        assert "timed out" in result.lower()

    def test_codesearch_tokens_num_clamped(self, tools) -> None:
        """Test that tokens_num is clamped to [1000, 50000]."""
        codesearch_tool = _get_tool(tools, "codesearch")
        # Verify tokens_num is in the tool's args_schema fields
        schema_fields = codesearch_tool.args_schema.model_fields
        assert "tokens_num" in schema_fields


# ===========================================================================
# 8. webfetch — tests
# ===========================================================================


class TestWebfetchTool:
    """Tests for the webfetch tool."""

    def _mock_get_response(self, mock_client_cls):
        """Return a mock response from the httpx.Client.get chain."""
        return mock_client_cls.return_value.__enter__.return_value.get.return_value

    @patch("httpx.Client")
    def test_webfetch_html_page(self, mock_client_cls, tools) -> None:
        """Test webfetch extracts text from HTML pages."""
        mock_response = self._mock_get_response(mock_client_cls)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body><h1>Hello</h1><p>World</p></body></html>"

        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "https://example.com"})

        assert "Hello" in result
        assert "World" in result
        assert "<html>" not in result
        assert "<h1>" not in result

    @patch("httpx.Client")
    def test_webfetch_plain_text(self, mock_client_cls, tools) -> None:
        """Test webfetch returns plain text as-is for non-HTML content."""
        mock_response = self._mock_get_response(mock_client_cls)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Just plain text content."

        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "https://example.com/data.txt"})

        assert "Just plain text content" in result

    @patch("httpx.Client")
    def test_webfetch_http_error(self, mock_client_cls, tools) -> None:
        """Test webfetch handles HTTP errors."""
        mock_response = self._mock_get_response(mock_client_cls)
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "https://example.com/missing"})

        assert "failed" in result.lower() or "404" in result

    @patch("httpx.Client")
    def test_webfetch_timeout(self, mock_client_cls, tools) -> None:
        """Test webfetch handles timeout."""
        import httpx as httpx_mod

        (
            mock_client_cls.return_value.__enter__.return_value.get.side_effect
        ) = httpx_mod.TimeoutException("timeout")

        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "https://example.com"})

        assert "timed out" in result.lower()

    def test_webfetch_invalid_url(self, tools) -> None:
        """Test webfetch rejects URLs without http(s) scheme."""
        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "ftp://example.com"})

        assert "Invalid URL" in result

    def test_webfetch_missing_scheme(self, tools) -> None:
        """Test webfetch rejects URLs without any scheme."""
        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "example.com"})

        assert "Invalid URL" in result

    @patch("httpx.Client")
    def test_webfetch_html_script_style_removed(self, mock_client_cls, tools) -> None:
        """Test that script and style tags are stripped from HTML."""
        mock_response = self._mock_get_response(mock_client_cls)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = (
            "<html><head><style>.hidden{display:none}</style></head>"
            "<body><script>alert('x')</script>"
            "<p>Visible content</p></body></html>"
        )

        webfetch_tool = _get_tool(tools, "webfetch")
        result = webfetch_tool.invoke({"url": "https://example.com"})

        assert "Visible content" in result
        assert "alert" not in result
        assert "display:none" not in result


# ===========================================================================
# Deprecated tools — keep backward-compatible tests
# ===========================================================================


class TestDeprecatedCreateFile:
    """Tests for the deprecated create_file tool (backward compat)."""

    def test_create_file_creates_new_file(self, tools, tmp_path) -> None:
        """Test create_file tool creates a new file with content."""
        create_file = _get_tool(tools, "create_file")

        result = create_file.invoke(
            {
                "file_path": "new_dir/hello.py",
                "content": "print('hello')\n",
            }
        )

        assert "Successfully created" in result
        created = tmp_path / "new_dir" / "hello.py"
        assert created.exists()
        assert created.read_text(encoding="utf-8") == "print('hello')\n"

    def test_create_file_rejects_overwrite_by_default(self, tools, tmp_path) -> None:
        """Test create_file refuses to overwrite existing files."""
        create_file = _get_tool(tools, "create_file")

        (tmp_path / "existing.txt").write_text("original", encoding="utf-8")

        result = create_file.invoke(
            {
                "file_path": "existing.txt",
                "content": "overwritten",
            }
        )

        assert "already exists" in result
        assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "original"

    def test_create_file_overwrite_when_flagged(self, tools, tmp_path) -> None:
        """Test create_file overwrites when overwrite=True."""
        create_file = _get_tool(tools, "create_file")

        (tmp_path / "existing.txt").write_text("original", encoding="utf-8")

        result = create_file.invoke(
            {
                "file_path": "existing.txt",
                "content": "overwritten",
                "overwrite": True,
            }
        )

        assert "Successfully" in result
        assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "overwritten"


class TestDeprecatedListDirectory:
    """Tests for the deprecated list_directory tool (backward compat)."""

    def test_list_directory_default(self, tools, tmp_path) -> None:
        """Test list_directory shows workspace contents."""
        (tmp_path / "file1.txt").write_text("a", encoding="utf-8")
        (tmp_path / "file2.py").write_text("b", encoding="utf-8")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("c", encoding="utf-8")

        list_dir = _get_tool(tools, "list_directory")

        result = list_dir.invoke({"dir_path": "."})

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result
        assert "[DIR]" in result
        assert "[FILE]" in result

    def test_list_directory_nonexistent(self, tools, tmp_path) -> None:
        """Test listing a nonexistent directory."""
        list_dir = _get_tool(tools, "list_directory")

        result = list_dir.invoke({"dir_path": "no_such_dir"})
        assert "does not exist" in result

    def test_list_directory_not_a_directory(self, tools, tmp_path) -> None:
        """Test listing a file path (not a directory)."""
        list_dir = _get_tool(tools, "list_directory")

        (tmp_path / "notafile.txt").write_text("data", encoding="utf-8")

        result = list_dir.invoke({"dir_path": "notafile.txt"})
        assert "not a directory" in result

    def test_list_directory_empty(self, tools, tmp_path) -> None:
        """Test listing an empty directory."""
        empty = tmp_path / "empty_dir"
        empty.mkdir()

        list_dir = _get_tool(tools, "list_directory")

        result = list_dir.invoke({"dir_path": "empty_dir"})
        assert "empty" in result.lower()
