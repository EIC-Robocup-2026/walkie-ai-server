#!/bin/bash
# scripts/run_app.sh — run the Flask app with uv (project venv / deps from pyproject.toml)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run python app.py "$@"
