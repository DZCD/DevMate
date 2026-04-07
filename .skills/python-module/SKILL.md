---
name: "python-module"
description: "Create a well-structured Python module with proper packaging, testing, and documentation"
---

# Python Module Template

Use this skill when creating a new Python module, package, or library.

## Recommended Structure

```
my_module/
  src/
    my_module/
      __init__.py
      core.py          # Main logic
      exceptions.py    # Custom exceptions
      types.py         # Type definitions
  tests/
    __init__.py
    test_core.py
    conftest.py
  pyproject.toml
  README.md
  CHANGELOG.md
```

## pyproject.toml Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-module"
version = "0.1.0"
description = "A brief description"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.8.0",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## Best Practices

- Use `src/` layout to avoid import confusion.
- All functions and classes must have docstrings (Google style).
- Use type hints for all function signatures.
- Use `logging` instead of `print()`.
- Use custom exception classes in `exceptions.py`.
- Keep `__init__.py` minimal — export only the public API.
- Write tests alongside code; aim for high coverage on core logic.
- Use `ruff` for linting and formatting; use `mypy` for type checking.
