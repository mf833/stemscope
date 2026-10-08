#!/usr/bin/env bash
# Start Stemscope on http://127.0.0.1:8765 and open it in the browser.
# The first run creates a virtualenv in .venv and installs the dependencies.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8765}"
URL="http://127.0.0.1:${PORT}/"

# (Re)build the virtualenv if it's missing or broken, e.g. after the folder was moved.
if ! .venv/bin/python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  PY="$(command -v python3.11 || command -v python3.12 || command -v python3.13 || command -v python3)"
  echo "Setting up the Python environment with ${PY} (one time, a few minutes)…"
  rm -rf .venv
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

# Open the browser once the server answers (torch takes a few seconds to import).
(
  for _ in $(seq 120); do
    if curl -s -o /dev/null "$URL"; then open "$URL"; exit 0; fi
    sleep 0.5
  done
) >/dev/null 2>&1 &

exec .venv/bin/python -m uvicorn backend.server:app --host 127.0.0.1 --port "$PORT"
