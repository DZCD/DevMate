"""Storage layer for DevMate.

Provides a Storage interface with file-based and in-memory implementations,
plus utilities for managing conversation message history.

Follows the architecture of agent-template-ts/src/storage/.
Uses local JSON files for persistence (zero external dependencies).
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default data directory for file storage
_DEFAULT_DATA_DIR = Path.home() / ".duclaw" / "devmate_data" / "memory"


# ---------------------------------------------------------------------------
# Storage interface (mirrors TS Storage<T>)
# ---------------------------------------------------------------------------


class Storage(ABC, Generic[T]):
    """Abstract storage interface with get/set/del operations."""

    @abstractmethod
    async def get(self, key: str) -> T | None: ...

    @abstractmethod
    async def set(self, key: str, value: T) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...


# ---------------------------------------------------------------------------
# File-based storage (primary implementation)
# ---------------------------------------------------------------------------


class FileStorage(Storage[T]):
    """File-based JSON storage implementation.

    Each key maps to a JSON file on disk.
    Directory structure: <base_dir>/<key>.json

    Follows agent-template-ts storage pattern with local file persistence.
    Zero external dependencies.
    """

    def __init__(self, base_dir: str | Path = _DEFAULT_DATA_DIR) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("FileStorage initialized at: %s", self._base_dir)

    def _file_path(self, key: str) -> Path:
        """Convert a storage key to a file path."""
        # Sanitize key to be a valid filename
        safe_key = key.replace(":", "_").replace("/", "_")
        return self._base_dir / f"{safe_key}.json"

    async def get(self, key: str) -> T | None:
        path = self._file_path(key)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return json.loads(text)  # type: ignore[return-value]
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("FileStorage get error for key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: T) -> None:
        path = self._file_path(key)
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
            # Sanitize surrogate characters from terminal input
            text = text.encode("utf-8", errors="surrogatepass").decode(
                "utf-8", errors="replace"
            )
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.error("FileStorage set error for key %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        path = self._file_path(key)
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.error("FileStorage delete error for key %s: %s", key, exc)


# ---------------------------------------------------------------------------
# In-memory storage (for testing)
# ---------------------------------------------------------------------------


class InMemoryStorage(Storage[T]):
    """In-memory dict-based storage implementation. Used as fallback."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> T | None:
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[return-value]

    async def set(self, key: str, value: T) -> None:
        self._store[key] = json.dumps(value, ensure_ascii=False)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


def create_storage(
    base_dir: str | Path | None = None,
) -> Storage[Any]:
    """Create a Storage instance.

    Uses FileStorage by default (local JSON files, zero external dependencies).
    Falls back to InMemoryStorage if base_dir is explicitly set to ":memory:".

    Args:
        base_dir: Directory for JSON files. If None, uses ~/.duclaw/devmate_data/memory/.
                  If ":memory:", uses InMemoryStorage.

    Returns:
        A Storage instance.
    """
    if base_dir == ":memory:":
        return InMemoryStorage()
    if base_dir is None:
        return FileStorage()
    return FileStorage(base_dir=base_dir)


# ---------------------------------------------------------------------------
# Message types (internal representation)
# ---------------------------------------------------------------------------


class TextBlock:
    """A text content block."""

    __slots__ = ("type", "text")

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


class ToolUseBlock:
    """A tool_use content block."""

    __slots__ = ("type", "id", "name", "input")

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:  # noqa: A002
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


class ToolResultBlock:
    """A tool_result content block."""

    __slots__ = ("type", "tool_use_id", "content")

    def __init__(self, tool_use_id: str, content: str) -> None:
        self.type = "tool_result"
        self.tool_use_id = tool_use_id
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class Message:
    """A conversation message."""

    __slots__ = ("role", "content", "reasoning_content")

    def __init__(
        self,
        role: str,
        content: str | list[ContentBlock],
        reasoning_content: str | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            d: dict[str, Any] = {"role": self.role, "content": self.content}
        else:
            d = {
                "role": self.role,
                "content": [
                    block.to_dict() if hasattr(block, "to_dict") else block
                    for block in self.content
                ],
            }
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Message":
        """Deserialize a Message from a dict."""
        role = data["role"]
        content = data["content"]
        reasoning_content = data.get("reasoning_content")
        if isinstance(content, str):
            return Message(role=role, content=content,
                           reasoning_content=reasoning_content)
        # Reconstruct content blocks
        blocks: list[ContentBlock] = []
        for block_data in content:
            block_type = block_data.get("type", "")
            if block_type == "text":
                blocks.append(TextBlock(text=block_data["text"]))
            elif block_type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=block_data["id"],
                        name=block_data["name"],
                        input=block_data.get("input", {}),
                    )
                )
            elif block_type == "tool_result":
                blocks.append(
                    ToolResultBlock(
                        tool_use_id=block_data["tool_use_id"],
                        content=block_data.get("content", ""),
                    )
                )
        return Message(role=role, content=blocks,
                       reasoning_content=reasoning_content)


def user_message(text: str) -> Message:
    """Create a user message with text content."""
    return Message(role="user", content=text)


def assistant_message(*blocks: ContentBlock) -> Message:
    """Create an assistant message with the given content blocks."""
    if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
        return Message(role="assistant", content=blocks[0].text)
    return Message(role="assistant", content=list(blocks))


def tool_result(tool_use_id: str, content: str) -> ToolResultBlock:
    """Create a tool_result block."""
    return ToolResultBlock(tool_use_id=tool_use_id, content=content)


# ---------------------------------------------------------------------------
# Message storage utilities (mirrors TS storage/utils.ts)
# ---------------------------------------------------------------------------


def _build_key(user_id: str, date: str | None = None) -> str:
    """Build the storage key for messages.

    Args:
        user_id: The user identifier.
        date: Optional date string in YYYYMMDD format.

    Returns:
        The storage key.
    """
    key = f"mem:{user_id}"
    if date:
        key = f"{key}:{date}"
    return key


async def get_messages(
    storage: Storage[Any],
    user_id: str,
    limit: int = 100,
    date: str | None = None,
) -> list[Message]:
    """Load message history for a user, respecting the message limit.

    Ensures the first message is a user message (OpenAI/Anthropic requirement).

    Args:
        storage: The storage backend.
        user_id: The user identifier.
        limit: Maximum number of messages to return.
        date: Optional date string in YYYYMMDD format.

    Returns:
        A list of Message objects, sanitized.
    """
    key = _build_key(user_id, date)
    data = await storage.get(key)
    if not data:
        return []

    # data is a list of dicts
    messages = [Message.from_dict(m) for m in data]
    if not messages:
        return []

    start = max(0, len(messages) - limit)

    # Ensure first message is user role
    if isinstance(messages[start].content, str) or (
        isinstance(messages[start].content, list)
        and any(b.type == "text" for b in messages[start].content if hasattr(b, "type"))
    ):
        if messages[start].role != "user":
            # Search backwards for a user message
            earlier = start - 1
            while earlier >= 0 and messages[earlier].role != "user":
                earlier -= 1
            if earlier >= 0:
                start = earlier
            else:
                # Search forwards
                later = start + 1
                while later < len(messages) and messages[later].role != "user":
                    later += 1
                if later >= len(messages):
                    return []
                start = later

    return sanitize_messages(messages[start:])


async def add_message(
    storage: Storage[Any],
    user_id: str,
    message: Message,
    date: str | None = None,
) -> int:
    """Append a message to the user's conversation history.

    Args:
        storage: The storage backend.
        user_id: The user identifier.
        message: The message to add.
        date: Optional date string in YYYYMMDD format.

    Returns:
        The index of the newly added message.
    """
    key = _build_key(user_id, date)
    data = await storage.get(key)
    if data is None:
        data = []
    data.append(message.to_dict())
    await storage.set(key, data)
    return len(data) - 1


def _message_has_tool_use(msg: Message) -> bool:
    return isinstance(msg.content, list) and any(
        isinstance(b, ToolUseBlock) for b in msg.content
    )


def _message_has_tool_result(msg: Message) -> bool:
    return isinstance(msg.content, list) and any(
        isinstance(b, ToolResultBlock) for b in msg.content
    )


def sanitize_messages(messages: list[Message]) -> list[Message]:
    """Sanitize message history to ensure API compliance.

    - Skips empty content messages
    - Ensures assistant tool_use messages have corresponding tool_result
    - Merges consecutive same-role messages, except across tool-call boundaries

    Mirrors TS storage/utils.ts sanitizeMessages().
    """
    if not messages:
        return messages

    result: list[Message] = []

    for i in range(len(messages)):
        msg = messages[i]

        # Skip empty content
        if isinstance(msg.content, str) and not msg.content.strip():
            continue
        if isinstance(msg.content, list) and len(msg.content) == 0:
            continue

        # Assistant message with tool_use: check next message has matching tool_result
        if msg.role == "assistant" and isinstance(msg.content, list):
            tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
            if tool_uses:
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if not next_msg or next_msg.role != "user":
                    logger.warning(
                        "[sanitize] assistant tool_use without tool_result, truncating at index=%d",
                        i,
                    )
                    break
                # Check all tool_use IDs have matching tool_result
                result_ids = set()
                if isinstance(next_msg.content, list):
                    for b in next_msg.content:
                        if isinstance(b, ToolResultBlock):
                            result_ids.add(b.tool_use_id)
                all_matched = all(tu.id in result_ids for tu in tool_uses)
                if not all_matched:
                    logger.warning(
                        "[sanitize] tool_use/tool_result ID mismatch, truncating at index=%d",
                        i,
                    )
                    break

        # Merge consecutive same-role messages, but preserve OpenAI tool-calling boundaries.
        prev = result[-1] if result else None
        if prev and prev.role == msg.role:
            prev_has_tool_use = _message_has_tool_use(prev)
            prev_has_tool_result = _message_has_tool_result(prev)
            msg_has_tool_use = _message_has_tool_use(msg)
            msg_has_tool_result = _message_has_tool_result(msg)

            if (
                prev_has_tool_use
                or prev_has_tool_result
                or msg_has_tool_use
                or msg_has_tool_result
            ):
                result.append(Message(
                    role=msg.role, content=msg.content,
                    reasoning_content=msg.reasoning_content,
                ))
                continue

            # Merge content only for plain same-role messages.
            if isinstance(prev.content, str) and isinstance(msg.content, str):
                prev = Message(
                    role=prev.role,
                    content=prev.content + "\n" + msg.content,
                )
                result[-1] = prev
            elif isinstance(prev.content, list) and isinstance(msg.content, list):
                merged_blocks = list(prev.content)
                merged_blocks.extend(msg.content)
                result[-1] = Message(
                    role=prev.role, content=merged_blocks,
                    reasoning_content=prev.reasoning_content,
                )
            elif isinstance(prev.content, str):
                # prev is str, msg is list
                blocks = [TextBlock(text=prev.content)]
                if isinstance(msg.content, list):
                    blocks.extend(msg.content)
                else:
                    blocks.append(TextBlock(text=msg.content))
                result[-1] = Message(
                    role=prev.role, content=blocks,
                    reasoning_content=prev.reasoning_content,
                )
            else:
                # prev is list, msg is str
                merged = list(prev.content)
                merged.append(TextBlock(text=msg.content))
                result[-1] = Message(
                    role=prev.role, content=merged,
                    reasoning_content=prev.reasoning_content,
                )
        else:
            result.append(Message(
                role=msg.role, content=msg.content,
                reasoning_content=msg.reasoning_content,
            ))

    return result


async def clear_messages(
    storage: Storage[Any],
    user_id: str,
    date: str | None = None,
) -> None:
    """Clear all messages for a user.

    Args:
        storage: The storage backend.
        user_id: The user identifier.
        date: Optional date string in YYYYMMDD format.
    """
    key = _build_key(user_id, date)
    await storage.delete(key)


def get_beijing_date() -> str:
    """Get the current date in Beijing timezone as YYYYMMDD."""
    from datetime import timedelta

    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime("%Y%m%d")
