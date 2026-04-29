#!/usr/bin/env bash
set -euo pipefail

URL="http://127.0.0.1:7777/sensor_health?n=${1:-24}"

echo "Checking sensor health: $URL"
resp="$(curl -fsS "$URL")"

RESP_JSON="$resp" /usr/bin/python3 - <<'PY'
import json
import os
import sys

data = json.loads(os.environ["RESP_JSON"])
overall = data.get("overall", "UNKNOWN")
print(f"Overall: {overall}")
print(f"Stale stream: {data.get('stale_stream')}")
print(f"Checked vectors: {data.get('checked_vectors')}")
print(f"Latest vector id: {data.get('latest_vector_id')}")
print("Sensors:")
for name, details in data.get("sensors", {}).items():
    print(f"- {name}: {details.get('status')} | {details}")

if overall == "FAIL":
    sys.exit(2)
if overall == "WARN":
    sys.exit(1)
PY
