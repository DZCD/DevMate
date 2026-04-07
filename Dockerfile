FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY config.toml .
COPY src/ src/
COPY mcp_server/ mcp_server/
COPY .skills/ .skills/
COPY docs/ docs/

# Install dependencies
RUN uv pip install --system -e .

# Expose MCP server port
EXPOSE 8001

# Default command: start MCP server
CMD ["python", "-m", "uvicorn", "mcp_server.server:main", "--host", "0.0.0.0", "--port", "8001"]
