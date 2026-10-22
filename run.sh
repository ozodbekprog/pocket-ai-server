#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
fi
if ! "$PY" -c "import fastapi, uvicorn, multipart" 2>/dev/null; then
  "$PY" -m pip install -q -r requirements.txt
fi
exec "$PY" server.py
