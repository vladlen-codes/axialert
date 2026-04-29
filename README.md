# Axialert

Name Origin: AxiAlert blends "Axi" (from axiom, meaning foundational truth) with "Alert," reflecting a reliable system that monitors patients and notifies caregivers of concerns.

## Architecture

```
ESP32 (sensor node)
  └─▶ POST /ingest  ──▶  ingestion_server.py
                              │
                         score inline (ML)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ThingsBoard                    POST /status back
  (telemetry + ML scores,           to ESP32 for LED /
   dashboard, alerting)              buzzer feedback
```

- **No local database.** All persistent storage and dashboarding is handled by ThingsBoard.
- **Inline scoring.** Each ingest request is scored immediately by the loaded ML models — no separate scorer process or delay.
- **Single process.** The full backend is one Flask server.

## Quick Start

### 1. Install dependencies

```bash
cd hardware/dementia-monitor-backend
python -m venv ../../.venv
source ../../.venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure ThingsBoard

```bash
cp .env.thingsboard.example .env.thingsboard
# Edit .env.thingsboard — fill in TB_ACCESS_TOKEN and optionally ESP32_URL
source .env.thingsboard
```

### 3. Start the server

```bash
./run_stack.sh
```

### 4. Verify it is running

```bash
./verify_stack.sh
```

### 5. Stop

```bash
./stop_stack.sh
```

## Live Endpoints

| Endpoint | Description |
|---|---|
| `POST http://0.0.0.0:7777/ingest` | Receive feature vector from ESP32 |
| `GET  http://127.0.0.1:7777/health` | Server health + feature flags |
| `GET  http://127.0.0.1:7777/status?n=20` | Last N scored vectors (in-memory) |

## Testing Without a Patient — Simulator

```bash
# Single scenario
python simulator.py --scenario normal_day
python simulator.py --scenario prolonged_inactivity
python simulator.py --scenario night_wandering
python simulator.py --scenario tachycardia_event
python simulator.py --scenario low_spo2_event

# All scenarios back-to-back (fast mode for CI)
python simulator.py --scenario all --fast

# Seed 2 days of synthetic data for model training
python simulator.py --bulk-days 2 --fast

# List all scenarios
python simulator.py --list
```

## Model Training

Run once after collecting real or synthetic data:

```bash
python train_models.py
```

Outputs trained models to `models/`. The server loads them automatically at startup.

## ThingsBoard Setup

See [thingsboard_setup.md](thingsboard_setup.md) for dashboard widget setup.

The ingestion server pushes the following telemetry keys per vector:

| Key | Description |
|---|---|
| `a_global` | Composite anomaly score `[0, 1]` |
| `anomaly_label` | e.g. `NORMAL`, `WARNING:NIGHT_WANDERING`, `CRITICAL:PHYSIOLOGICAL_EVENT` |
| `severity` | `NORMAL` / `WARNING` / `CRITICAL` (for widget colour rules) |
| `hr_mean` / `spo2_mean` | Latest vitals |
| `pir_ratio` | Motion activity ratio |
| `app_on` / `app_on_s` | Appliance state |

## Notes

- The `.venv` Python is at `/path/to/axialert/.venv/bin/python` — the run script detects it automatically.
- Logs are written to `hardware/dementia-monitor-backend/logs/`.
- `secrets.h` (firmware Wi-Fi credentials) is gitignored — never committed.
