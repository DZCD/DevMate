"""LLM abstraction layer for DevMate.

Provides LLMClient interface and OpenAI-compatible adapter.

Follows the architecture of agent-template-ts/src/llm/.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from devmate.storage import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LLMToolDef:
    """Tool definition in the format expected by the LLM API."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: list[ContentBlock]
    finish_reason: str  # "stop", "tool_calls", "length"
    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLMClient interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client interface.

    Mirrors agent-template-ts/src/llm/LLMClient.ts.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[LLMToolDef] | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (for DeepSeek, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleAdapter(LLMClient):
    """Adapter for OpenAI-compatible APIs (DeepSeek, etc.).

    Converts internal Message/ContentBlock types to/from OpenAI API format.
    Mirrors agent-template-ts/src/llm/AnthropicAdapter.ts.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI API format."""
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    openai_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg.content, list):
                    parts: list[dict[str, Any]] = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parts.append({"type": "text", "text": block.text})
                        elif isinstance(block, ToolResultBlock):
                            parts.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.tool_use_id,
                                    "content": block.content,
                                }
                            )
                    openai_messages.append({"role": "user", "content": parts})
                else:
                    openai_messages.append(
                        {"role": "user", "content": str(msg.content)}
                    )

            elif msg.role == "assistant":
                # Check if this assistant message has tool_use blocks
                has_tool_use = False
                if isinstance(msg.content, list):
                    has_tool_use = any(isinstance(b, ToolUseBlock) for b in msg.content)

                if has_tool_use:
                    # Extract text and tool_use parts
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []

                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(
                                            block.input, ensure_ascii=False
                                        ),
                                    },
                                }
                            )

                    msg_dict: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        msg_dict["content"] = "\n".join(text_parts)
                    else:
                        msg_dict["content"] = None
                    msg_dict["tool_calls"] = tool_calls
                    openai_messages.append(msg_dict)
                else:
                    # Simple text message
                    if isinstance(msg.content, str):
                        openai_messages.append(
                            {"role": "assistant", "content": msg.content}
                        )
                    elif isinstance(msg.content, list):
                        text = "\n".join(
                            b.text for b in msg.content if isinstance(b, TextBlock)
                        )
                        openai_messages.append({"role": "assistant", "content": text})
                    else:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": str(msg.content),
                            }
                        )
            else:
                # Fallback for other roles
                openai_messages.append(
                    {
                        "role": msg.role,
                        "content": str(msg.content)
                        if not isinstance(msg.content, str)
                        else msg.content,
                    }
                )

        return openai_messages

    def _to_openai_tools(self, tools: list[LLMToolDef]) -> list[dict[str, Any]]:
        """Convert LLMToolDef to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI API response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"

        content_blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        # Text content
        if message.content:
            content_blocks.append(TextBlock(text=message.content))

        # Tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_id = tc.id or str(uuid.uuid4())
                func = tc.function
                try:
                    args = json.loads(func.arguments) if func.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content_blocks.append(
                    ToolUseBlock(id=tool_id, name=func.name, input=args)
                )
                tool_calls.append(ToolCall(id=tool_id, name=func.name, arguments=args))

        # Map finish reasons
        if finish_reason == "tool_calls":
            mapped_reason = "tool_calls"
        elif finish_reason == "length":
            mapped_reason = "length"
        else:
            mapped_reason = "stop"

        return LLMResponse(
            content=content_blocks,
            finish_reason=mapped_reason,
            tool_calls=tool_calls,
        )

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[LLMToolDef] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI-compatible API."""
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(self._to_openai_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        try:
            response = await self._client.chat.completions.create(**kwargs)
            return self._parse_response(response)
        except Exception as exc:
            logger.error("[OpenAICompatibleAdapter] API call failed: %s", exc)
            raise
