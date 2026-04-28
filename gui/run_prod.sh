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

# ── build frontend ───────────────────────────────────────────────────────────
echo "▶  Installing frontend dependencies…"
conda run -n torch3090 npm install --prefix "$SCRIPT_DIR/frontend"

echo "▶  Building frontend…"
conda run -n torch3090 npm run --prefix "$SCRIPT_DIR/frontend" build

echo "▶  Starting production server on :$PORT"
conda run -n torch3090 \
  python -m uvicorn gui.backend.main:app \
  --host 0.0.0.0 --port "$PORT" \
  --app-dir "$REPO_ROOT"
