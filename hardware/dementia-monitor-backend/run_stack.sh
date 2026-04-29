#!/usr/bin/env bash
# run_stack.sh — Start the Axialert ingestion server (ThingsBoard-first, no local DB)
#
# SETUP (one time):
#   1. Copy .env.thingsboard.example → .env.thingsboard
#   2. Fill in TB_ACCESS_TOKEN (and optionally ESP32_URL)
#   3. source .env.thingsboard
#   4. ./run_stack.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect Python (repo venv preferred, system python3 as fallback)
VENV_PYTHON="$ROOT_DIR/../../.venv/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="$(command -v python3)"
  echo "[WARN] .venv not found — using system python3: $PYTHON_BIN"
else
  echo "[ERROR] No Python found. Create .venv or install python3."
  exit 1
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ingestion_server.log"
PORT=7777

# Check if already running
if lsof -i ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  EXISTING_PID="$(lsof -i ":${PORT}" -sTCP:LISTEN -t | head -n 1)"
  echo "[SKIP] ingestion_server already listening on :${PORT} (pid ${EXISTING_PID})"
  exit 0
fi

# Warn if ThingsBoard token not set
if [[ -z "${TB_ACCESS_TOKEN:-}" ]]; then
  echo "[WARN] TB_ACCESS_TOKEN not set — ThingsBoard push disabled."
  echo "       Run: source .env.thingsboard  to enable it."
fi

echo "[START] ingestion_server"
nohup "$PYTHON_BIN" "$ROOT_DIR/ingestion_server.py" >"$LOG_FILE" 2>&1 &
PID=$!
sleep 1

if kill -0 "$PID" >/dev/null 2>&1; then
  echo "[OK]   ingestion_server started (pid ${PID})"
else
  echo "[FAIL] ingestion_server did not start. Check $LOG_FILE"
  exit 1
fi

echo ""
echo "Stack running."
echo "  Ingest  : http://127.0.0.1:${PORT}/ingest"
echo "  Health  : http://127.0.0.1:${PORT}/health"
echo "  Recent  : http://127.0.0.1:${PORT}/status?n=20"
echo "  Log     : $LOG_FILE"
if [[ -n "${TB_ACCESS_TOKEN:-}" ]]; then
  echo "  ThingsBoard: https://${TB_HOST:-demo.thingsboard.io}"
fi
