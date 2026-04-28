#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── load .env ────────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "⚠  $ENV_FILE not found. Copy .env.example to .env and fill in your paths."
  exit 1
fi

PORT="${PORT:-8000}"

# ── start backend ────────────────────────────────────────────────────────────
echo "▶  Starting backend on :$PORT"
conda run -n torch3090 \
  python -m uvicorn gui.backend.main:app \
  --host 0.0.0.0 --port "$PORT" --reload \
  --app-dir "$REPO_ROOT" &
BACKEND_PID=$!

# ── install frontend deps if needed ─────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
  echo "▶  Installing frontend dependencies…"
  conda run -n torch3090 npm install --prefix "$SCRIPT_DIR/frontend"
fi

# ── start frontend dev server ────────────────────────────────────────────────
echo "▶  Starting frontend dev server on :5173"
trap "kill $BACKEND_PID 2>/dev/null; exit 0" INT TERM
conda run -n torch3090 npm run --prefix "$SCRIPT_DIR/frontend" dev
