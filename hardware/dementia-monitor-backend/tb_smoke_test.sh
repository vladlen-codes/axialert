#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="/Users/vlad/Documents/GitHub/axialert/.venv/bin/python"

if [[ -f "$ROOT_DIR/.env.thingsboard" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.thingsboard"
fi

if [[ -z "${TB_ACCESS_TOKEN:-}" ]]; then
  echo "[FAIL] TB_ACCESS_TOKEN is not set. Source .env.thingsboard first."
  exit 1
fi

echo "[1/5] Check ingestion health"
curl -fsS http://127.0.0.1:7777/health >/dev/null

echo "[2/5] Read current latest vector id"
BEFORE_ID="$(curl -fsS "http://127.0.0.1:7777/vectors?n=1" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["id"] if d else 0)')"
echo "      before_id=$BEFORE_ID"

echo "[3/5] POST one synthetic telemetry vector"
TS_NOW="$(date +%s)"
HOUR_NOW="$(date +%H)"
PAYLOAD="{\"ts\":${TS_NOW},\"hour\":${HOUR_NOW},\"hour_sin\":0.0,\"hour_cos\":1.0,\"pir_count\":4,\"pir_ratio\":0.12,\"no_motion_s\":20.0,\"cur_mean\":2.2,\"cur_std\":0.07,\"app_on\":1,\"app_on_s\":40.0,\"app_switches\":2,\"hr_mean\":74.0,\"hr_std\":1.3,\"spo2_mean\":97.8,\"spo2_std\":0.4,\"tachy\":0,\"brady\":0,\"spo2_low\":0,\"vitals_valid\":1,\"mot_and_app\":1,\"mot_no_app\":0,\"nomot_abnvit\":0}"
POST_RESP="$(curl -fsS -X POST http://127.0.0.1:7777/ingest -H "Content-Type: application/json" -d "$PAYLOAD")"
echo "      ingest_response=$POST_RESP"

echo "[4/5] Verify new vector was stored"
AFTER_ID="$(curl -fsS "http://127.0.0.1:7777/vectors?n=1" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["id"] if d else 0)')"
echo "      after_id=$AFTER_ID"
if [[ "$AFTER_ID" -le "$BEFORE_ID" ]]; then
  echo "[FAIL] Vector id did not increase."
  exit 1
fi

echo "[5/5] Run ThingsBoard bridge one-shot sync"
SYNC_OUT="$($PYTHON_BIN "$ROOT_DIR/thingsboard_bridge.py" --once 2>&1)"
echo "$SYNC_OUT"

if [[ "$SYNC_OUT" == *"Synced vector #$AFTER_ID"* ]]; then
  echo "[PASS] ThingsBoard smoke test succeeded (vector #$AFTER_ID synced)."
else
  echo "[WARN] Bridge ran, but did not report expected vector id #$AFTER_ID."
  exit 1
fi
