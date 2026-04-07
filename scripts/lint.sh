#!/bin/bash
# PEP 8 / code quality checks for DevMate.
#
# Usage:
#   bash scripts/lint.sh          # check source + tests
#   bash scripts/lint.sh --fix    # auto-fix formatting issues
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

ERRORS=0

echo "=== DevMate Lint Checks ==="
echo ""

# ---------------------------------------------------------------------------
# 1. ruff format
# ---------------------------------------------------------------------------
echo "[1/4] ruff format check..."
if [ "$FIX_MODE" = true ]; then
    if ruff format src/ mcp_server/ tests/; then
        echo "  ruff format: FIXED"
    else
        echo "  ruff format: FIX FAILED"
        ERRORS=$((ERRORS + 1))
    fi
else
    if ruff format --check src/ mcp_server/ tests/; then
        echo "  ruff format: PASS"
    else
        echo "  ruff format: FAIL (run with --fix to auto-fix)"
        ERRORS=$((ERRORS + 1))
    fi
fi

# ---------------------------------------------------------------------------
# 2. ruff lint (E = pycodestyle errors, W = warnings, F = pyflakes, I = isort)
# ---------------------------------------------------------------------------
echo "[2/4] ruff lint check..."
if [ "$FIX_MODE" = true ]; then
    if ruff check --fix src/ mcp_server/ tests/; then
        echo "  ruff lint: FIXED"
    else
        echo "  ruff lint: FIX FAILED (some issues need manual fix)"
        ERRORS=$((ERRORS + 1))
    fi
else
    if ruff check --select E,W,F,I src/ mcp_server/ tests/; then
        echo "  ruff lint: PASS"
    else
        echo "  ruff lint: FAIL"
        ERRORS=$((ERRORS + 1))
    fi
fi

# ---------------------------------------------------------------------------
# 3. print() ban in source code
# ---------------------------------------------------------------------------
echo "[3/4] print() ban check..."
# Only look at .py files, skip __pycache__
# Match lines that look like actual print() calls (not inside strings)
PRINT_HITS=$(grep -rn --include="*.py" '^\s*print\s*(' src/devmate/ mcp_server/ 2>/dev/null || true)
if [ -n "$PRINT_HITS" ]; then
    echo "  print() ban: FAIL"
    echo "  Found print() calls:"
    echo "$PRINT_HITS"
    ERRORS=$((ERRORS + 1))
else
    echo "  print() ban: PASS"
fi

# ---------------------------------------------------------------------------
# 4. Imports check – ensure no relative imports outside packages
# ---------------------------------------------------------------------------
echo "[4/4] import style check..."
# Check for any 'from devmate.agent import ...' that uses relative syntax
# incorrectly (e.g. from .agent import) outside the package
BAD_IMPORTS=$(grep -rn --include="*.py" "^from \.\." src/devmate/ mcp_server/ tests/ 2>/dev/null || true)
if [ -n "$BAD_IMPORTS" ]; then
    echo "  import style: WARN (double-dot relative imports found)"
    echo "$BAD_IMPORTS"
    # Not a hard failure, just a warning
else
    echo "  import style: PASS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "=== All checks passed! ==="
    exit 0
else
    echo "=== $ERRORS check(s) FAILED ==="
    exit 1
fi
