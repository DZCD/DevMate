# DevMate Project Template Guide

This document provides the standard project template and conventions for new projects within the DevMate ecosystem. It is specifically designed for internal web development projects.

## Recommended Project Structure

```
project-name/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection setup
│   ├── models/              # SQLAlchemy / Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routers/             # API route modules
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── services/            # Business logic layer
│   │   ├── __init__.py
│   │   └── service.py
│   ├── static/              # Static assets
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/           # Jinja2 / HTML templates
│       └── base.html
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── migrations/              # Alembic migrations (if applicable)
├── docs/                    # Project documentation
├── pyproject.toml
├── .gitignore
├── .python-version
├── README.md
└── Dockerfile
```

## Tech Stack

### Backend
- **Framework**: FastAPI (latest stable)
- **Database**: PostgreSQL with SQLAlchemy 2.0 async
- **Validation**: Pydantic v2
- **Migration**: Alembic
- **Testing**: pytest + pytest-asyncio + httpx

### Frontend
- **Templates**: Jinja2
- **Styling**: Tailwind CSS (via CDN or build)
- **Interactivity**: Vanilla JS or Alpine.js (keep it lightweight)

### DevOps
- **Container**: Docker + Docker Compose
- **Python Management**: uv
- **Linting**: ruff
- **Formatting**: ruff format

## Frontend Template Guidelines

### HTML Base Template

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Project{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2/dist/tailwind.min.css" rel="stylesheet">
    {% block extra_css %}{% endblock %}
</head>
<body class="bg-gray-50">
    <nav class="bg-white shadow">
        <div class="max-w-7xl mx-auto px-4 py-3">
            <a href="/" class="text-xl font-bold text-blue-600">Project</a>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>

    <footer class="bg-gray-800 text-white py-6 mt-12">
        <div class="max-w-7xl mx-auto px-4 text-center text-sm">
            &copy; 2024 Project Team
        </div>
    </footer>

    {% block extra_js %}{% endblock %}
</body>
</html>
```

### Key Frontend Conventions
- All pages must extend the base template
- Use semantic HTML5 elements
- Ensure mobile responsiveness (use Tailwind responsive classes)
- Keep JavaScript minimal and use Alpine.js for interactivity
- All text content should support Chinese localization

## Example Scenario: Hiking Website

For a hiking/outdoor website project, follow these specific patterns:

### Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime

class HikingRoute(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    distance_km: float = Field(..., gt=0)
    elevation_gain: int = Field(..., ge=0)
    description: str
    image_url: str | None = None
    created_at: datetime

class HikingRouteCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    distance_km: float = Field(..., gt=0)
    elevation_gain: int = Field(..., ge=0)
    description: str
    image_url: str | None = None
```

### API Routes

```python
from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api/routes", tags=["hiking"])

@router.get("/", response_model=list[HikingRoute])
async def list_routes(difficulty: str | None = None):
    """List all hiking routes with optional difficulty filter."""
    routes = await route_service.get_all(difficulty=difficulty)
    return routes

@router.get("/{route_id}", response_model=HikingRoute)
async def get_route(route_id: int):
    """Get a specific hiking route by ID."""
    route = await route_service.get_by_id(route_id)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route
```

## Database Conventions

1. Use snake_case for table and column names
2. Always include `created_at` and `updated_at` timestamps
3. Use UUID for primary keys when data may be synced externally
4. Add appropriate indexes for frequently queried columns
5. Use soft deletes (is_deleted flag) for important entities

## Testing Strategy

1. **Unit tests**: Test individual functions and services
2. **Integration tests**: Test API endpoints with test database
3. **E2E tests**: Test critical user flows (optional, use Playwright)

```python
# conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
```

## Deployment Checklist

- [ ] All tests pass (`uv run pytest`)
- [ ] No ruff linting errors (`uv run ruff check .`)
- [ ] Code is formatted (`uv run ruff format --check .`)
- [ ] Environment variables documented in README
- [ ] Docker image builds successfully
- [ ] Database migrations are up to date
- [ ] Health check endpoint (`/health`) is available
