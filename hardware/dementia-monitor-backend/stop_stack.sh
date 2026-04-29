#!/usr/bin/env bash
set -euo pipefail

if pkill -f "ingestion_server.py" 2>/dev/null; then
  echo "[STOP] ingestion_server"
else
  echo "[SKIP] ingestion_server not running"
fi

echo "Done."
