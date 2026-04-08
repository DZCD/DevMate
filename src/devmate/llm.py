"""LLM abstraction layer for DevMate.

Provides LLMClient interface and a ChatOpenAI-based adapter.

Uses ``langchain_openai.ChatOpenAI`` as the underlying LLM driver, converting
between internal Message / ContentBlock types and LangChain message types.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

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
# ChatOpenAI adapter
# ---------------------------------------------------------------------------


class OpenAICompatibleAdapter(LLMClient):
    """Adapter using ``langchain_openai.ChatOpenAI`` for LLM calls.

    Converts internal Message / ContentBlock types to / from LangChain
    message types so the rest of the codebase stays unchanged.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> None:
        self._llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._model = model

    # -- message conversion --------------------------------------------------

    @staticmethod
    def _to_langchain_messages(
        messages: list[Message],
        system_prompt: str,
    ) -> list[Any]:
        """Convert internal messages to LangChain message objects."""
        lc_messages: list[Any] = [
            SystemMessage(content=system_prompt),
        ]

        for msg in messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    lc_messages.append(
                        HumanMessage(content=msg.content)
                    )
                elif isinstance(msg.content, list):
                    text_parts: list[str] = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolResultBlock):
                            lc_messages.append(
                                ToolMessage(
                                    content=block.content,
                                    tool_call_id=block.tool_use_id,
                                )
                            )
                    if text_parts:
                        lc_messages.append(
                            HumanMessage(
                                content="\n".join(text_parts)
                            )
                        )
                else:
                    lc_messages.append(
                        HumanMessage(content=str(msg.content))
                    )

            elif msg.role == "assistant":
                if isinstance(msg.content, list):
                    tool_calls: list[dict[str, Any]] = []
                    text_parts: list[str] = []

                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "name": block.name,
                                    "args": block.input,
                                }
                            )

                    lc_messages.append(
                        AIMessage(
                            content="\n".join(text_parts) if text_parts else "",
                            tool_calls=tool_calls or None,
                        )
                    )
                else:
                    lc_messages.append(
                        AIMessage(
                            content=str(msg.content)
                            if not isinstance(msg.content, str)
                            else msg.content
                        )
                    )
            else:
                lc_messages.append(
                    HumanMessage(content=str(msg.content))
                )

        return lc_messages

    @staticmethod
    def _to_langchain_tools(
        tools: list[LLMToolDef],
    ) -> list[dict[str, Any]]:
        """Convert LLMToolDef list to OpenAI function-calling format."""
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

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Parse LangChain AIMessage into internal LLMResponse."""
        content_blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        # Text content
        text = response.content
        if text and isinstance(text, str) and text.strip():
            content_blocks.append(TextBlock(text=text))
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, str) and part.strip():
                    content_blocks.append(TextBlock(text=part))
                elif isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if txt.strip():
                        content_blocks.append(TextBlock(text=txt))

        # Tool calls
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        for tc in raw_tool_calls:
            tool_id = tc.get("id", str(uuid.uuid4()))
            name = tc.get("name", "")
            args = tc.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            content_blocks.append(
                ToolUseBlock(id=tool_id, name=name, input=args)
            )
            tool_calls.append(
                ToolCall(id=tool_id, name=name, arguments=args)
            )

        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        # Check for length truncation via response_metadata
        resp_meta = getattr(response, "response_metadata", {}) or {}
        finish = resp_meta.get("finish_reason", "")
        token_usage = resp_meta.get("token_usage", {}) or {}
        if finish == "length":
            finish_reason = "length"
        elif (
            isinstance(token_usage, dict)
            and token_usage.get("completion_tokens_details")
        ):
            details = token_usage["completion_tokens_details"]
            if (
                isinstance(details, dict)
                and details.get("reasoning_tokens", 0) > 0
            ):
                pass  # normal completion with reasoning

        return LLMResponse(
            content=content_blocks,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    # -- public interface ----------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[LLMToolDef] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via ChatOpenAI."""
        lc_messages = self._to_langchain_messages(messages, system_prompt)

        llm = self._llm
        if tools:
            lc_tools = self._to_langchain_tools(tools)
            llm = llm.bind_tools(lc_tools)

        try:
            response = await llm.ainvoke(lc_messages)
            return self._parse_response(response)
        except Exception as exc:
            logger.error("[ChatOpenAIAdapter] API call failed: %s", exc)
            raise
