"""Image understanding tool for DevMate.

Uses a vision-capable LLM (e.g. kimi-k2.5) to analyze images.
Supports local file paths and HTTP URLs.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _guess_media_type(file_path: str) -> str:
    """Guess image media type from file extension."""
    suffix = Path(file_path).suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mapping.get(suffix, "image/png")


def _guess_media_type_from_base64(data: str) -> str:
    """Guess image media type from base64 data magic bytes."""
    if data.startswith("/9j/"):
        return "image/jpeg"
    if data.startswith("iVBOR"):
        return "image/png"
    if data.startswith("R0lG"):
        return "image/gif"
    if data.startswith("UklGR"):
        return "image/webp"
    return "image/png"


async def _resolve_image(source: str) -> tuple[str, str]:
    """Resolve image source to (base64_data, media_type).

    Args:
        source: A local file path or HTTP(S) URL.

    Returns:
        Tuple of (base64_encoded_data, media_type).
    """
    if source.startswith("http://") or source.startswith("https://"):
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(source)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            data = base64.b64encode(response.content).decode("utf-8")
            if "jpeg" in content_type or "jpg" in content_type:
                media_type = "image/jpeg"
            elif "png" in content_type:
                media_type = "image/png"
            elif "gif" in content_type:
                media_type = "image/gif"
            elif "webp" in content_type:
                media_type = "image/webp"
            else:
                media_type = _guess_media_type_from_base64(data)
            return data, media_type

    # Local file path
    path = Path(source)
    logger.debug("Checking image path: %s (absolute: %s)", source, path.absolute())
    if not path.exists():
        msg = f"Image file not found: {source} (cwd: {Path.cwd()})"
        logger.error(msg)
        raise FileNotFoundError(msg)
    try:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        media_type = _guess_media_type(source)
        logger.info("Successfully loaded image: %s (%d bytes)", source, len(data))
        return data, media_type
    except Exception as e:
        logger.error("Failed to read image file %s: %s", source, e)
        raise


def create_image_understand_tool(
    api_key: str,
    base_url: str,
    model: str,
) -> Any:
    """Create an image understanding tool using a vision-capable LLM.

    Args:
        api_key: API key for the vision model.
        base_url: Base URL for the vision model API.
        model: Model name (must support vision/image input).

    Returns:
        A LangChain tool function.
    """

    @tool
    async def image_understand(image_path: str, prompt: str) -> str:
        """Analyze and understand the content of an image file.

        Use this tool when you need to:
        - Describe what's in an image
        - Extract text from screenshots
        - Analyze UI/UX designs from mockups
        - Understand charts, diagrams, or photos
        - Get design inspiration from reference images

        Args:
            image_path: Path to a local image file or HTTP(S) URL.
            prompt: What you want to know about the image,
                e.g. "Describe the layout and design style".
        """
        logger.info("Analyzing image: %s", image_path[:80])

        try:
            base64_data, media_type = await _resolve_image(image_path)
        except FileNotFoundError as e:
            logger.error("Image file not found: %s", e)
            return f"Error: {e}"
        except Exception as e:
            logger.error("Failed to resolve image %s: %s", image_path, e)
            return f"Error loading image: {type(e).__name__}: {e}"

        image_url = f"data:{media_type};base64,{base64_data}"

        # Call vision-capable LLM using OpenAI-compatible format
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                return f"Image analysis failed: {response.status_code} - {response.text}"
            data = response.json()

        content = data["choices"][0]["message"].get("content", "")
        logger.info("Image analysis complete (%d chars)", len(content))
        return content

    return image_understand
