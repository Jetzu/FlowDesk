#!/usr/bin/env bash
# Launch FlowDesk on macOS or Linux (creates the virtual environment on first run).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
  else
    PY=python3
    command -v python3.12 >/dev/null 2>&1 && PY=python3.12
    "$PY" -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e .
  fi
fi
exec .venv/bin/python -m flowdesk "$@"
