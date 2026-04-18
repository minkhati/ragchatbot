#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCES="backend/ main.py"

check_format() {
    echo "--- Checking formatting (black) ---"
    uv run black --check $SOURCES
}

fix_format() {
    echo "--- Formatting with black ---"
    uv run black $SOURCES
}

run_tests() {
    echo "--- Running tests (pytest) ---"
    cd backend && uv run pytest tests/ -v
}

case "${1:-check}" in
    check)
        check_format
        echo ""
        echo "All quality checks passed."
        ;;
    fix)
        fix_format
        echo ""
        echo "Formatting applied."
        ;;
    test)
        run_tests
        ;;
    all)
        check_format
        echo ""
        run_tests
        echo ""
        echo "All checks passed."
        ;;
    *)
        echo "Usage: $0 [check|fix|test|all]"
        echo "  check  - verify formatting (default)"
        echo "  fix    - auto-format all files"
        echo "  test   - run pytest"
        echo "  all    - check formatting + run tests"
        exit 1
        ;;
esac
