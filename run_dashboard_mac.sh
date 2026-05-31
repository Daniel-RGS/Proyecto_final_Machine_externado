#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${STREAMLIT_PORT:-8501}"

cd "$PROJECT_ROOT"

if command -v lsof >/dev/null 2>&1; then
  ORIGINAL_PORT="$PORT"
  while lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
    PORT=$((PORT + 1))
  done
  if [[ "$PORT" != "$ORIGINAL_PORT" ]]; then
    echo "Port $ORIGINAL_PORT is already in use; using $PORT instead."
  fi
fi

"$PYTHON_BIN" -m streamlit run dashboard/app.py --server.port "$PORT"
