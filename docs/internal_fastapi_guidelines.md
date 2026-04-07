# Internal FastAPI Development Guidelines

## Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point
│   ├── routers/         # API route modules
│   ├── models/          # Pydantic models
│   ├── services/        # Business logic
│   ├── dependencies.py  # Dependency injection
│   └── config.py        # Configuration
├── tests/
└── pyproject.toml
```

## API Design Principles

### RESTful Conventions

1. Use plural nouns for resource endpoints: `/users`, `/items`
2. Use HTTP methods correctly:
   - GET: Retrieve resources
   - POST: Create resources
   - PUT: Update resources (full replacement)
   - PATCH: Partial updates
   - DELETE: Remove resources
3. Use proper HTTP status codes:
   - 200: Success
   - 201: Created
   - 204: No Content
   - 400: Bad Request
   - 401: Unauthorized
   - 403: Forbidden
   - 404: Not Found
   - 422: Validation Error
   - 500: Internal Server Error

### Request/Response Models

Always use Pydantic models for request bodies and responses:

```python
from pydantic import BaseModel, Field
from datetime import datetime

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    price: float = Field(..., gt=0)

class ItemResponse(BaseModel):
    id: int
    name: str
    description: str | None
    price: float
    created_at: datetime

    model_config = {"from_attributes": True}
```

### Dependency Injection

Use FastAPI's dependency injection system:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

@router.get("/items/{item_id}")
async def get_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
):
    ...
```

## Database

### SQLAlchemy Async

Use SQLAlchemy 2.0 with async support:

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass
```

### Migration

Use Alembic for database migrations. Keep migrations in version control.

## Error Handling

Create custom exception handlers:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
```

## Testing

Use `httpx.AsyncClient` with `pytest-asyncio` for testing:

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_create_item():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/items", json={"name": "Test", "price": 10.0})
    assert response.status_code == 201
```

## Performance

1. Use `async def` for all route handlers that do I/O
2. Use connection pooling for databases
3. Implement caching with `fastapi-cache2` or Redis
4. Use pagination for list endpoints
5. Add proper indexes to database tables
