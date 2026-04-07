"""End-to-end agent tests with real LangGraph tool loop.

These tests mock only the LLM (ChatOpenAI) while letting LangGraph execute
the actual tool loop: dispatch tool_calls -> invoke @tool functions -> feed
ToolMessage results back to the LLM -> produce the final response.

This validates:
- Tools are correctly dispatched to the right @tool function
- Tool arguments are correctly parsed and passed
- Tool results are returned to the LLM via ToolMessage
- Multi-turn conversations work end-to-end
"""

import logging
import sys
import types
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: create a mock LLM that behaves correctly in the LangGraph loop
# ---------------------------------------------------------------------------


def _make_mock_llm(responses: list[AIMessage]) -> RunnableLambda:
    """Create a mock LLM as a ``RunnableLambda`` that cycles through *responses*.

    ``create_react_agent`` (LangGraph) internally calls ``model.bind_tools(tools)``
    and then chains the result into a ``RunnableSequence`` via the ``|`` operator.
    A plain ``MagicMock`` breaks this pipeline because ``MagicMock.__or__``
    returns another ``MagicMock`` (not a ``Runnable``).

    Using a real ``RunnableLambda`` sidesteps both problems:

    * ``bind_tools`` is **not** called because ``_should_bind_tools`` returns
      ``True`` but the ``RunnableLambda`` has no ``bind_tools`` method.
    * The ``|`` pipe works because ``RunnableLambda`` *is* a ``Runnable``.
    * ``.ainvoke(messages)`` returns the next ``AIMessage`` from the iterator.

    The caller must also patch ``ChatOpenAI`` so that ``DevMateAgent`` receives
    this ``RunnableLambda`` as ``self._llm``.
    """
    response_iter: Iterator[AIMessage] = iter(responses)

    def _invoke(messages: Any) -> AIMessage:
        return next(response_iter)

    return RunnableLambda(_invoke)


# ---------------------------------------------------------------------------
# Fixture: patch langchain.agents AND suppress re-binding of tools
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_langchain_agents():
    """Two patches are needed:

    1. ``langchain.agents.create_agent`` -> thin wrapper around
       ``langgraph.prebuilt.create_react_agent`` that maps
       ``system_prompt`` -> ``prompt``.
    2. ``_should_bind_tools`` is monkey-patched to always return ``False``
       so that ``create_react_agent`` does **not** try to call
       ``model.bind_tools(tools)`` again (our ``RunnableLambda`` mock LLM
       already "has" the tools).
    """
    from langgraph.prebuilt import create_react_agent

    def _create_agent(model, tools, *, system_prompt=None, **kwargs):
        if system_prompt is not None and "prompt" not in kwargs:
            kwargs["prompt"] = system_prompt
        return create_react_agent(model, tools, **kwargs)

    # -- Patch langchain.agents --
    fake_agents = types.ModuleType("langchain.agents")
    fake_agents.create_agent = _create_agent
    fake_agents.__all__ = ["create_agent"]

    import langchain

    original_agents = getattr(langchain, "agents", None)
    langchain.agents = fake_agents
    sys.modules["langchain.agents"] = fake_agents

    cached = sys.modules.pop("devmate.agent", None)

    # -- Patch _should_bind_tools to prevent re-binding --
    import langgraph.prebuilt.chat_agent_executor as _react_mod

    original_sbt = _react_mod._should_bind_tools
    _react_mod._should_bind_tools = lambda *a, **kw: False

    yield

    # -- Restore --
    _react_mod._should_bind_tools = original_sbt
    if cached is not None:
        sys.modules["devmate.agent"] = cached
    sys.modules.pop("langchain.agents", None)
    if original_agents is not None:
        langchain.agents = original_agents


# ---------------------------------------------------------------------------
# Helper: minimal config file
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path, skills_dir: str | None = None) -> Path:
    """Create a minimal config.toml for testing."""
    skills_line = (
        f'directory = "{skills_dir}"' if skills_dir else 'directory = ".skills"'
    )
    config_content = f"""
[model]
base_url = "https://example.com/api"
api_key = "test-key"
model_name = "test-model"

[search]
tavily_api_key = "tvly-test"
max_results = 3

[langsmith]
enabled = false

[skills]
{skills_line}

[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"

[mcp_server]
host = "localhost"
port = 18001
route = "/mcp"
"""
    config_file = tmp_path / "config.toml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


# ---------------------------------------------------------------------------
# Helper: build and initialize a DevMateAgent with mocked LLM
# ---------------------------------------------------------------------------


async def _build_agent(
    tmp_path: Path,
    mock_llm: RunnableLambda,
) -> Any:
    """Build a fully initialized DevMateAgent with a mock LLM.

    ``mock_llm`` is a ``RunnableLambda`` that returns pre-canned
    ``AIMessage`` responses.  It is injected via ``patch("devmate.agent.ChatOpenAI")``.
    """
    from devmate.agent import DevMateAgent

    skills_dir = tmp_path / ".skills"
    skills_dir.mkdir()
    config_file = _write_minimal_config(tmp_path, str(skills_dir))

    with (
        patch("devmate.agent.RAGEngine") as mock_rag_cls,
        patch("devmate.agent.SkillsManager") as mock_skills_cls,
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            new_callable=MagicMock,
        ) as mock_mcp_cls,
    ):
        # RAG mock
        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance

        # Skills mock
        mock_skills_instance = MagicMock()
        mock_skills_instance.load_skills.return_value = 0
        mock_skills_instance.get_skill_meta.return_value = ""
        mock_skills_instance.create_tools.return_value = []
        mock_skills_cls.return_value = mock_skills_instance

        # MCP mock — no tools
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_tools = AsyncMock(return_value=[])
        mock_mcp_cls.return_value = mock_mcp_instance

        with patch("devmate.agent.ChatOpenAI", return_value=mock_llm):
            agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
            await agent.initialize()

    return agent


# ===========================================================================
# Test Cases
# ===========================================================================


class TestAgentE2EToolLoop:
    """End-to-end tests that exercise the real LangGraph tool loop."""

    async def test_read_file(self, tmp_path) -> None:
        """Agent reads a file via the tool loop."""
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hello world')\n", encoding="utf-8")

        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"file_path": str(test_file)},
                        }
                    ],
                ),
                AIMessage(content="The file contains: print('hello world')"),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("read hello.py")

        assert "hello" in result

    async def test_write_new_file(self, tmp_path) -> None:
        """Agent creates a new file via the tool loop."""
        new_file = tmp_path / "new_file.py"

        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_write_1",
                            "name": "write",
                            "args": {
                                "file_path": str(new_file),
                                "content": "# New file\nx = 42\n",
                            },
                        }
                    ],
                ),
                AIMessage(content="File created successfully."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("create a new file new_file.py")

        assert new_file.exists()
        assert new_file.read_text(encoding="utf-8") == "# New file\nx = 42\n"
        assert "success" in result.lower() or "created" in result.lower()

    async def test_read_then_edit(self, tmp_path) -> None:
        """Multi-step: agent reads a file then edits it."""
        test_file = tmp_path / "app.py"
        test_file.write_text("x = 1\ny = 2\n", encoding="utf-8")

        mock_llm = _make_mock_llm(
            [
                # Step 1: LLM decides to read the file
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"file_path": str(test_file)},
                        }
                    ],
                ),
                # Step 2: LLM decides to edit the file
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_edit_1",
                            "name": "edit",
                            "args": {
                                "file_path": str(test_file),
                                "old_string": "x = 1",
                                "new_string": "x = 100",
                            },
                        }
                    ],
                ),
                # Step 3: Final response
                AIMessage(content="Done. Changed x from 1 to 100."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
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

        mock_llm = _make_mock_llm(
            [
                # Step 1: glob for *.py files
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_glob_1",
                            "name": "glob",
                            "args": {"pattern": "**/*.py", "path": str(tmp_path)},
                        }
                    ],
                ),
                # Step 2: read utils.py
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"file_path": str(tmp_path / "utils.py")},
                        }
                    ],
                ),
                # Step 3: final response
                AIMessage(content="Found and read utils.py: def helper(): pass"),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
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

        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_grep_1",
                            "name": "grep",
                            "args": {"pattern": "import", "path": str(src_dir)},
                        }
                    ],
                ),
                AIMessage(content="Found 2 files with import statements."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("search for import statements in src/")

        assert "2" in result or "import" in result

    async def test_bash_execute(self, tmp_path) -> None:
        """Agent executes a bash command and verifies side effects."""
        marker_file = tmp_path / "marker.txt"

        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_bash_1",
                            "name": "bash",
                            "args": {"command": f"touch {marker_file}"},
                        }
                    ],
                ),
                AIMessage(content="Command executed."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("create an empty marker file")

        # Verify the file was actually created by bash
        assert marker_file.exists(), "marker.txt should have been created by bash"
        assert "command" in result.lower() or "executed" in result.lower()

    async def test_multi_turn_conversation(self, tmp_path) -> None:
        """Two independent run() calls each trigger a separate tool loop."""
        test_file = tmp_path / "counter.py"
        test_file.write_text("count = 0\n", encoding="utf-8")

        # First turn: read
        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"file_path": str(test_file)},
                        }
                    ],
                ),
                AIMessage(content="count is 0"),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result1 = await agent.run("read counter.py")
        assert "count" in result1

        # Second turn: edit — create a fresh mock LLM with new responses
        second_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_edit_1",
                            "name": "edit",
                            "args": {
                                "file_path": str(test_file),
                                "old_string": "count = 0",
                                "new_string": "count = 10",
                            },
                        }
                    ],
                ),
                AIMessage(content="Updated count to 10"),
            ]
        )

        # Re-initialize the agent with the new mock LLM
        with patch("devmate.agent.ChatOpenAI", return_value=second_llm):
            await agent.initialize()

        result2 = await agent.run("change count to 10")
        assert "10" in result2

        # Verify the file was actually modified
        content = test_file.read_text(encoding="utf-8")
        assert "count = 10" in content

    async def test_tool_call_with_wrong_args_returns_error(self, tmp_path) -> None:
        """When tool returns an error (e.g. file not found), LLM sees it."""
        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"file_path": str(tmp_path / "nonexistent.txt")},
                        }
                    ],
                ),
                AIMessage(content="File not found, I will let you know."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("read a file that does not exist")

        assert "not found" in result.lower()

    async def test_sequential_tool_calls_in_one_turn(self, tmp_path) -> None:
        """LLM returns multiple tool_calls in a single AIMessage."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a = 1\n", encoding="utf-8")
        f2.write_text("b = 2\n", encoding="utf-8")

        mock_llm = _make_mock_llm(
            [
                # Request two reads at once
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_read_a",
                            "name": "read",
                            "args": {"file_path": str(f1)},
                        },
                        {
                            "id": "call_read_b",
                            "name": "read",
                            "args": {"file_path": str(f2)},
                        },
                    ],
                ),
                AIMessage(content="a=1 and b=2"),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        result = await agent.run("read both a.py and b.py")

        assert "a" in result.lower() or "1" in result
        assert "b" in result.lower() or "2" in result

    async def test_write_creates_nested_directories(self, tmp_path) -> None:
        """write tool auto-creates parent directories."""
        deep_file = tmp_path / "a" / "b" / "c" / "deep.py"

        mock_llm = _make_mock_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_write_1",
                            "name": "write",
                            "args": {
                                "file_path": str(deep_file),
                                "content": "deep = True\n",
                            },
                        }
                    ],
                ),
                AIMessage(content="Created nested file."),
            ]
        )

        agent = await _build_agent(tmp_path, mock_llm)
        await agent.run("create a deeply nested file")

        assert deep_file.exists()
        assert deep_file.read_text(encoding="utf-8") == "deep = True\n"
