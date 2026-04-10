"""LLM abstraction layer for DevMate.

Uses ``httpx`` directly so that model-specific fields
(e.g. kimi's ``reasoning_content``) are fully preserved across rounds.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from langsmith.run_helpers import traceable

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
    reasoning_content: str | None = None


# ---------------------------------------------------------------------------
# LLMClient interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[LLMToolDef] | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (raw httpx)
# ---------------------------------------------------------------------------


class OpenAICompatibleAdapter(LLMClient):
    """Adapter using raw httpx for LLM calls.

    Bypasses the OpenAI SDK so model-specific fields like
    ``reasoning_content`` are fully captured from the JSON response.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    # -- message conversion --------------------------------------------------

    @staticmethod
    def _to_raw_messages(
        messages: list[Message],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Convert internal messages to raw OpenAI API dict format."""
        raw: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        for msg in messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    raw.append({"role": "user", "content": msg.content})
                elif isinstance(msg.content, list):
                    parts: list[str] = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                        elif isinstance(block, ToolResultBlock):
                            raw.append(
                                {
                                    "role": "tool",
                                    "content": block.content,
                                    "tool_call_id": block.tool_use_id,
                                }
                            )
                    if parts:
                        raw.append(
                            {"role": "user", "content": "\n".join(parts)}
                        )
                else:
                    raw.append({"role": "user", "content": str(msg.content)})

            elif msg.role == "assistant":
                msg_dict: dict[str, Any] = {"role": "assistant"}

                # Preserve reasoning_content for thinking-mode models
                if msg.reasoning_content is not None:
                    msg_dict["reasoning_content"] = msg.reasoning_content

                if isinstance(msg.content, list):
                    text_parts: list[str] = []
                    tool_calls_raw: list[dict[str, Any]] = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls_raw.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                }
                            )
                    msg_dict["content"] = (
                        "\n".join(text_parts) if text_parts else None
                    )
                    if tool_calls_raw:
                        msg_dict["tool_calls"] = tool_calls_raw
                else:
                    msg_dict["content"] = str(msg.content)

                raw.append(msg_dict)
            else:
                raw.append({"role": msg.role, "content": str(msg.content)})

        return raw

    @staticmethod
    def _to_openai_tools(
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

    # -- response parsing ----------------------------------------------------

    @staticmethod
    def _parse_response_json(data: dict[str, Any]) -> LLMResponse:
        """Parse raw JSON response dict into internal LLMResponse."""
        choice = data["choices"][0]
        message = choice["message"]

        content_blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        text = message.get("content")
        if text and isinstance(text, str) and text.strip():
            content_blocks.append(TextBlock(text=text))

        for tc in message.get("tool_calls") or []:
            func = tc.get("function", {})
            args: dict[str, Any] = {}
            raw_args = func.get("arguments")
            if raw_args:
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            content_blocks.append(
                ToolUseBlock(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    input=args,
                )
            )
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )

        finish_reason = choice.get("finish_reason", "stop") or "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        reasoning_content = message.get("reasoning_content")

        return LLMResponse(
            content=content_blocks,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    # -- public interface ----------------------------------------------------

    @traceable(name="llm_chat", run_type="llm")
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[LLMToolDef] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via raw httpx."""
        raw_messages = self._to_raw_messages(messages, system_prompt)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": raw_messages,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if tools:
            payload["tools"] = self._to_openai_tools(tools)

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                url, headers=headers, json=payload,
            )
            if response.status_code != 200:
                logger.error(
                    "[LLM] API error %d: %s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(
                    f"API error {response.status_code}: {response.text}"
                )

            data = response.json()
            result = self._parse_response_json(data)

            logger.info(
                "LLM response: finish_reason=%s, tool_calls=%d, "
                "reasoning_content=%s",
                result.finish_reason,
                len(result.tool_calls),
                "present" if result.reasoning_content else "None",
            )

            return result
        except Exception as exc:
            logger.error("[LLM] API call failed: %s", exc)
            raise
