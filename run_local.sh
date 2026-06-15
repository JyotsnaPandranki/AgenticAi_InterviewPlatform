#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/pranavrajivreddy/Hackathon"
FRONTEND_DIR="$ROOT_DIR/frontend"

cd "$ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "venv not found. Create it first: python3 -m venv venv"
  exit 1
fi

source venv/bin/activate

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "[HIA] Starting backend on :8000 ..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[HIA] Starting frontend on :5173 ..."
cd "$FRONTEND_DIR"
VITE_API_BASE_URL=http://localhost:8000 npm run dev &
FRONTEND_PID=$!

echo "[HIA] Running. Open: http://localhost:5173"
wait

