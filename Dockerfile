FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files first (for dependency caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache -e .

# Copy application source code
COPY src/ src/
COPY mcp_server/ mcp_server/
COPY .skills/ .skills/
COPY docs/ docs/

# Config is mounted at runtime, not baked into the image
COPY config.toml.example config.toml.example

# Expose MCP server port
EXPOSE 8001

# Default command: start MCP server
# Uses module-level entry point so uvicorn can find the ASGI app
CMD ["python", "-m", "mcp_server.server"]
