#!/usr/bin/env bash
set -euo pipefail

ok=1

echo "[CHECK] ingestion health"
HEALTH=$(curl -fsS http://127.0.0.1:7777/health 2>/dev/null || echo "FAIL")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
  echo "  PASS"
  # Show key flags from health response
  echo "  models_loaded   : $(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('models_loaded'))")"
  echo "  thingsboard     : $(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('thingsboard'))")"
  echo "  esp32_feedback  : $(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('esp32_feedback'))")"
else
  echo "  FAIL (server not responding)"
  ok=0
fi

echo "[CHECK] process"
if pgrep -f "ingestion_server.py" >/dev/null 2>&1; then
  echo "  PASS ingestion_server.py"
else
  echo "  FAIL ingestion_server.py not running"
  ok=0
fi

if [[ "$ok" -eq 1 ]]; then
  echo "RUNNING"
  exit 0
else
  echo "NOT RUNNING"
  exit 1
fi
