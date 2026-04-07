"""Tests for the custom tool loop mechanism.

Tests the core tool loop logic in isolation: LLM response parsing,
tool dispatching, message history management, and sanitization.
"""

from unittest.mock import AsyncMock

import pytest

from devmate.storage import (
    FileStorage,
    InMemoryStorage,
    Message,
    add_message,
    get_messages,
    sanitize_messages,
    user_message,
)
from devmate.storage import (
    ToolUseBlock as StorageToolUseBlock,
)
from devmate.tools import Tool, ToolExecutor, ToolRegistry, tools_to_llm_defs

# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestInMemoryStorage:
    """Tests for the in-memory storage implementation."""

    async def test_get_set_del(self) -> None:
        """Test basic get/set/del operations."""
        storage = InMemoryStorage()

        result = await storage.get("key1")
        assert result is None

        await storage.set("key1", {"data": "hello"})
        result = await storage.get("key1")
        assert result == {"data": "hello"}

        await storage.delete("key1")
        result = await storage.get("key1")
        assert result is None

    async def test_overwrite(self) -> None:
        """Test that set overwrites existing values."""
        storage = InMemoryStorage()
        await storage.set("key1", "value1")
        await storage.set("key1", "value2")
        result = await storage.get("key1")
        assert result == "value2"


class TestFileStorage:
    """Tests for the file-based storage implementation."""

    async def test_get_set_del(self, tmp_path) -> None:
        """Test basic get/set/del operations with FileStorage."""
        storage = FileStorage(base_dir=tmp_path / "storage")

        result = await storage.get("key1")
        assert result is None

        await storage.set("key1", {"data": "hello"})
        result = await storage.get("key1")
        assert result == {"data": "hello"}

        await storage.delete("key1")
        result = await storage.get("key1")
        assert result is None

    async def test_overwrite(self, tmp_path) -> None:
        """Test that set overwrites existing values."""
        storage = FileStorage(base_dir=tmp_path / "storage")
        await storage.set("key1", "value1")
        await storage.set("key1", "value2")
        result = await storage.get("key1")
        assert result == "value2"

    async def test_persistence(self, tmp_path) -> None:
        """Test that data persists across storage instances."""
        storage1 = FileStorage(base_dir=tmp_path / "storage")
        await storage1.set("persist_key", [1, 2, 3])

        storage2 = FileStorage(base_dir=tmp_path / "storage")
        result = await storage2.get("persist_key")
        assert result == [1, 2, 3]

    async def test_creates_directory(self, tmp_path) -> None:
        """Test that FileStorage creates the base directory."""
        nested = tmp_path / "a" / "b" / "c"
        FileStorage(base_dir=nested)
        assert nested.exists()
        assert nested.is_dir()

    async def test_key_sanitization(self, tmp_path) -> None:
        """Test that keys with special characters are sanitized."""
        storage = FileStorage(base_dir=tmp_path / "storage")
        await storage.set("mem:user:20260407", "data")
        # Check the file was created with sanitized name
        files = list((tmp_path / "storage").glob("*.json"))
        assert len(files) == 1
        assert "mem_user_20260407" in files[0].name


class TestSanitizeMessages:
    """Tests for the sanitize_messages function."""

    def test_empty_list(self) -> None:
        """Test empty message list."""
        result = sanitize_messages([])
        assert result == []

    def test_merge_consecutive_same_role(self) -> None:
        """Test merging consecutive same-role messages."""
        messages = [
            user_message("hello"),
            user_message("world"),
        ]
        result = sanitize_messages(messages)
        assert len(result) == 1
        assert result[0].role == "user"

    def test_preserve_alternating_roles(self) -> None:
        """Test that alternating roles are preserved."""
        messages = [
            user_message("hello"),
            Message(role="assistant", content="hi there"),
        ]
        result = sanitize_messages(messages)
        assert len(result) == 2

    def test_skip_empty_messages(self) -> None:
        """Test that empty messages are skipped."""
        messages = [
            user_message("hello"),
            user_message(""),
        ]
        result = sanitize_messages(messages)
        assert len(result) == 1

    def test_truncate_on_unmatched_tool_use(self) -> None:
        """Test truncation when assistant tool_use has no matching tool_result."""
        messages = [
            user_message("hello"),
            Message(
                role="assistant",
                content=[
                    StorageToolUseBlock(id="tc_1", name="read", input={}),
                ],
            ),
            # Missing tool_result - next message is assistant
            Message(role="assistant", content="next"),
        ]
        result = sanitize_messages(messages)
        # Should truncate at the unmatched tool_use
        assert len(result) == 1
        assert result[0].role == "user"


class TestMessageStorage:
    """Tests for message storage utilities."""

    async def test_add_and_get_messages(self) -> None:
        """Test adding and retrieving messages."""
        storage = InMemoryStorage()
        await add_message(storage, "user1", user_message("hello"), "20260101")
        await add_message(
            storage,
            "user1",
            Message(role="assistant", content="hi there"),
            "20260101",
        )

        messages = await get_messages(storage, "user1", date="20260101")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    async def test_empty_storage(self) -> None:
        """Test getting messages from empty storage."""
        storage = InMemoryStorage()
        messages = await get_messages(storage, "user1")
        assert messages == []

    async def test_limit_messages(self) -> None:
        """Test that message limit is respected."""
        storage = InMemoryStorage()
        for i in range(10):
            await add_message(storage, "user1", user_message(f"msg {i}"), "20260101")
            await add_message(
                storage,
                "user1",
                Message(role="assistant", content=f"resp {i}"),
                "20260101",
            )

        messages = await get_messages(storage, "user1", limit=4, date="20260101")
        assert len(messages) <= 4


# ---------------------------------------------------------------------------
# Tool Registry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for the tool registry."""

    async def _make_tool(self, name: str) -> Tool:
        """Create a simple tool for testing."""

        async def execute(**kwargs: object) -> str:
            return f"{name} executed with {kwargs}"

        return Tool(
            name=name,
            description=f"A test tool: {name}",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )

    async def test_register_and_get(self) -> None:
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = await self._make_tool("test_tool")
        registry.register(tool)

        assert registry.has("test_tool")
        assert registry.get("test_tool") is tool
        assert len(registry) == 1

    async def test_register_duplicate_raises(self) -> None:
        """Test that registering a duplicate tool raises ValueError."""
        registry = ToolRegistry()
        tool1 = await self._make_tool("dup")
        tool2 = await self._make_tool("dup")
        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    async def test_get_all(self) -> None:
        """Test getting all tools."""
        registry = ToolRegistry()
        tools = [await self._make_tool(f"tool_{i}") for i in range(3)]
        for t in tools:
            registry.register(t)

        all_tools = registry.get_all()
        assert len(all_tools) == 3


class TestToolExecutor:
    """Tests for the tool executor."""

    async def test_execute_tool(self) -> None:
        """Test executing a tool."""
        registry = ToolRegistry()

        async def execute(**kwargs: object) -> str:
            return "tool executed"

        tool = Tool(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        registry.register(tool)
        executor = ToolExecutor(registry)

        result = await executor.execute("test", {})
        assert result == "tool executed"

    async def test_execute_nonexistent_tool(self) -> None:
        """Test executing a non-existent tool returns error."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = await executor.execute("nonexistent", {})
        assert "not found" in result

    async def test_execute_missing_required_param(self) -> None:
        """Test that missing required parameters returns error."""
        registry = ToolRegistry()

        async def execute(**kwargs: object) -> str:
            return "ok"

        tool = Tool(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            execute=execute,
        )
        registry.register(tool)
        executor = ToolExecutor(registry)

        result = await executor.execute("test", {})
        assert "missing required" in result

    async def test_execute_tool_error_caught(self) -> None:
        """Test that tool execution errors are caught and returned."""
        registry = ToolRegistry()

        async def execute(**kwargs: object) -> str:
            raise RuntimeError("tool error")

        tool = Tool(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        registry.register(tool)
        executor = ToolExecutor(registry)

        result = await executor.execute("test", {})
        assert "execution failed" in result


# ---------------------------------------------------------------------------
# LLMToolDef conversion tests
# ---------------------------------------------------------------------------


class TestToolConversion:
    """Tests for tool conversion utilities."""

    def test_tools_to_llm_defs(self) -> None:
        """Test converting tools to LLM tool definitions."""
        tools = [
            Tool(
                name="test1",
                description="Test tool 1",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                execute=AsyncMock(),
            ),
            Tool(
                name="test2",
                description="Test tool 2",
                parameters={"type": "object", "properties": {"y": {"type": "number"}}},
                execute=AsyncMock(),
            ),
        ]

        defs = tools_to_llm_defs(tools)
        assert len(defs) == 2
        assert defs[0].name == "test1"
        assert defs[1].name == "test2"
