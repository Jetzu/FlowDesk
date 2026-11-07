#!/usr/bin/env bash
# Launch FlowDesk on macOS or Linux (creates the virtual environment on first run).
# Usage: ./flowdesk.sh [workspace.fdw | file.fcs | folder ...]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Make relative file arguments absolute before changing directory.
ARGS=()
for a in "$@"; do
  if [ -e "$a" ] && [ "${a#/}" = "$a" ]; then ARGS+=("$PWD/$a"); else ARGS+=("$a"); fi
done
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
  else
    PY=""
    for c in python3.12 python3.13 python3.11 python3; do
      if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
    if [ -z "$PY" ]; then
      echo "Python 3.11+ not found. Install python3 (Linux: sudo apt install python3 python3-venv) or uv." >&2
      exit 1
    fi
    "$PY" -m venv .venv || { echo "venv creation failed (Debian/Ubuntu: sudo apt install python3-venv)" >&2; rm -rf .venv; exit 1; }
    .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -e .
  fi
fi
exec .venv/bin/python -m flowdesk "${ARGS[@]}"
