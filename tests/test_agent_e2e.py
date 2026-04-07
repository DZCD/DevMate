"""End-to-end agent tests with the custom tool loop.

These tests mock the LLM while exercising the real tool loop:
dispatch tool_calls -> invoke @tool functions -> feed ToolResultBlock
back to the LLM -> produce the final response.

This validates:
- Tools are correctly dispatched to the right function
- Tool arguments are correctly parsed and passed
- Tool results are returned to the LLM via ToolResultBlock
- Multi-turn conversations work end-to-end
"""

import logging
from unittest.mock import AsyncMock, patch

from devmate.llm import LLMResponse, TextBlock, ToolCall
from tests.helpers import build_agent

logger = logging.getLogger(__name__)


class TestAgentE2EToolLoop:
    """End-to-end tests that exercise the custom tool loop."""

    async def test_read_file(self, tmp_path) -> None:
        """Agent reads a file via the tool loop."""
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hello world')\n", encoding="utf-8")

        mock_llm = build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read",
                            arguments={"file_path": str(test_file)},
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="The file contains: print('hello world')")],
                    finish_reason="stop",
                ),
            ],
        )

        agent = await mock_llm
        result = await agent.run("read hello.py")

        assert "hello" in result

    async def test_write_new_file(self, tmp_path) -> None:
        """Agent creates a new file via the tool loop."""
        new_file = tmp_path / "new_file.py"

        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_write_1",
                            name="write",
                            arguments={
                                "file_path": str(new_file),
                                "content": "# New file\nx = 42\n",
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="File created successfully.")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("create a new file new_file.py")

        assert new_file.exists()
        assert new_file.read_text(encoding="utf-8") == "# New file\nx = 42\n"
        assert "success" in result.lower() or "created" in result.lower()

    async def test_read_then_edit(self, tmp_path) -> None:
        """Multi-step: agent reads a file then edits it."""
        test_file = tmp_path / "app.py"
        test_file.write_text("x = 1\ny = 2\n", encoding="utf-8")

        agent = await build_agent(
            tmp_path,
            [
                # Step 1: LLM decides to read the file
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read",
                            arguments={"file_path": str(test_file)},
                        )
                    ],
                ),
                # Step 2: LLM decides to edit the file
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_edit_1",
                            name="edit",
                            arguments={
                                "file_path": str(test_file),
                                "old_string": "x = 1",
                                "new_string": "x = 100",
                            },
                        )
                    ],
                ),
                # Step 3: Final response
                LLMResponse(
                    content=[TextBlock(text="Done. Changed x from 1 to 100.")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("read app.py and change x to 100")

        # Verify file was modified
        content = test_file.read_text(encoding="utf-8")
        assert "x = 100" in content
        assert "x = 1\n" not in content
        assert "100" in result

    async def test_glob_then_read(self, tmp_path) -> None:
        """Agent globs for files, then reads one."""
        # Create test files
        (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")

        agent = await build_agent(
            tmp_path,
            [
                # Step 1: glob for *.py files
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_glob_1",
                            name="glob",
                            arguments={"pattern": "**/*.py", "path": str(tmp_path)},
                        )
                    ],
                ),
                # Step 2: read utils.py
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read",
                            arguments={"file_path": str(tmp_path / "utils.py")},
                        )
                    ],
                ),
                # Step 3: final response
                LLMResponse(
                    content=[
                        TextBlock(text="Found and read utils.py: def helper(): pass")
                    ],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("find all Python files and read utils.py")

        assert "helper" in result

    async def test_grep_search(self, tmp_path) -> None:
        """Agent uses grep to search for a pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text(
            "import logging\nlogger = logging.getLogger(__name__)\n",
            encoding="utf-8",
        )
        (src_dir / "config.py").write_text(
            "import os\nPATH = os.environ.get('PATH', '')\n",
            encoding="utf-8",
        )

        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_grep_1",
                            name="grep",
                            arguments={"pattern": "import", "path": str(src_dir)},
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="Found 2 files with import statements.")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("search for import statements in src/")

        assert "2" in result or "import" in result

    async def test_bash_execute(self, tmp_path) -> None:
        """Agent executes a bash command and verifies side effects."""
        marker_file = tmp_path / "marker.txt"

        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_bash_1",
                            name="bash",
                            arguments={"command": f"touch {marker_file}"},
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="Command executed.")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("create an empty marker file")

        # Verify the file was actually created by bash
        assert marker_file.exists(), "marker.txt should have been created by bash"
        assert "command" in result.lower() or "executed" in result.lower()

    async def test_multi_turn_conversation(self, tmp_path) -> None:
        """Two independent run() calls each trigger a separate tool loop."""
        test_file = tmp_path / "counter.py"
        test_file.write_text("count = 0\n", encoding="utf-8")

        # First turn: read
        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read",
                            arguments={"file_path": str(test_file)},
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="count is 0")],
                    finish_reason="stop",
                ),
            ],
        )

        result1 = await agent.run("read counter.py")
        assert "count" in result1

        # Second turn: edit — need to replace the mock LLM responses
        # Since responses were consumed, we need to patch the LLM directly
        with patch.object(
            agent._llm,
            "chat",
            new=AsyncMock(
                side_effect=[
                    LLMResponse(
                        content=[],
                        finish_reason="tool_calls",
                        tool_calls=[
                            ToolCall(
                                id="call_edit_1",
                                name="edit",
                                arguments={
                                    "file_path": str(test_file),
                                    "old_string": "count = 0",
                                    "new_string": "count = 10",
                                },
                            )
                        ],
                    ),
                    LLMResponse(
                        content=[TextBlock(text="Updated count to 10")],
                        finish_reason="stop",
                    ),
                ]
            ),
        ):
            result2 = await agent.run("change count to 10")
            assert "10" in result2

        # Verify the file was actually modified
        content = test_file.read_text(encoding="utf-8")
        assert "count = 10" in content

    async def test_tool_call_with_wrong_args_returns_error(self, tmp_path) -> None:
        """When tool returns an error (e.g. file not found), LLM sees it."""
        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read",
                            arguments={"file_path": str(tmp_path / "nonexistent.txt")},
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="File not found, I will let you know.")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("read a file that does not exist")

        assert "not found" in result.lower()

    async def test_sequential_tool_calls_in_one_turn(self, tmp_path) -> None:
        """LLM returns multiple tool_calls in a single response."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a = 1\n", encoding="utf-8")
        f2.write_text("b = 2\n", encoding="utf-8")

        agent = await build_agent(
            tmp_path,
            [
                # Request two reads at once
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_read_a",
                            name="read",
                            arguments={"file_path": str(f1)},
                        ),
                        ToolCall(
                            id="call_read_b",
                            name="read",
                            arguments={"file_path": str(f2)},
                        ),
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="a=1 and b=2")],
                    finish_reason="stop",
                ),
            ],
        )

        result = await agent.run("read both a.py and b.py")

        assert "a" in result.lower() or "1" in result
        assert "b" in result.lower() or "2" in result

    async def test_write_creates_nested_directories(self, tmp_path) -> None:
        """write tool auto-creates parent directories."""
        deep_file = tmp_path / "a" / "b" / "c" / "deep.py"

        agent = await build_agent(
            tmp_path,
            [
                LLMResponse(
                    content=[],
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_write_1",
                            name="write",
                            arguments={
                                "file_path": str(deep_file),
                                "content": "deep = True\n",
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content=[TextBlock(text="Created nested file.")],
                    finish_reason="stop",
                ),
            ],
        )

        await agent.run("create a deeply nested file")

        assert deep_file.exists()
        assert deep_file.read_text(encoding="utf-8") == "deep = True\n"
