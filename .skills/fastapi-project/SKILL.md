---
name: "fastapi-project"
description: "Create a FastAPI project with recommended structure, dependencies, and best practices"
---

# FastAPI Project Template

Use this skill when creating a new FastAPI project or adding features to an existing one.

## Project Structure

```
project_root/
  app/
    __init__.py
    main.py          # FastAPI application entry point
    config.py        # Settings via pydantic-settings
    models/          # Pydantic models / SQLAlchemy models
    schemas/         # Request/response schemas
    routers/         # API route modules
    services/        # Business logic layer
    dependencies.py  # Shared dependencies
  tests/
    conftest.py
    test_*.py
  pyproject.toml
  README.md
```

## Quick Start

1. Create the project directory structure above.
2. Add dependencies to `pyproject.toml`:

```toml
[project]
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "ruff>=0.8.0",
]
```

## main.py Template

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="My API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Best Practices

- Use async handlers for I/O-bound operations.
- Organize routes into `routers/` modules and include them via `app.include_router()`.
- Use dependency injection for shared resources (DB sessions, auth).
- Return Pydantic models, not raw dicts.
- Use `pydantic-settings` for configuration management.
- Add OpenAPI tags for route grouping.
- Write tests with `httpx.AsyncClient` and pytest-asyncio.
