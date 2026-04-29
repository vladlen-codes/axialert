# Axialert — Technical Documentation
> Dementia Monitoring Platform | ESP32 + ML Backend + Real-Time Dashboard

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [ESP32 Firmware — How It Works](#3-esp32-firmware--how-it-works)
4. [Feature Vector](#4-feature-vector)
5. [Backend — Ingestion Server](#5-backend--ingestion-server)
6. [ML Pipeline & Algorithms](#6-ml-pipeline--algorithms)
7. [Anomaly Scoring & Fusion](#7-anomaly-scoring--fusion)
8. [Anomaly Classification Rules](#8-anomaly-classification-rules)
9. [Feedback Loop (ESP32 ↔ Backend)](#9-feedback-loop-esp32--backend)
10. [ThingsBoard Integration](#10-thingsboard-integration)
11. [Dashboard](#11-dashboard)
12. [Simulator](#12-simulator)
13. [Model Training](#13-model-training)
14. [Alert Severity Summary](#14-alert-severity-summary)
15. [Data Flow Diagram](#15-data-flow-diagram)

---

## 1. System Overview

**Axialert** (from *axiom* + *alert*) is a real-time dementia patient monitoring system that fuses:

- **Motion detection** (PIR sensor) — tracks activity and inactivity patterns
- **Appliance usage** (ACS712 current sensor) — infers ADL (Activities of Daily Living)
- **Physiological vitals** (MAX30102 pulse-oximeter) — measures heart rate & SpO₂

Data is collected on an **ESP32 microcontroller**, transmitted as structured feature vectors to a **Python/Flask backend**, scored by three ML models, and displayed on a **real-time web dashboard**. Alerts are pushed back to the ESP32 for physical LED/buzzer feedback, and to **ThingsBoard** for cloud dashboarding.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ESP32 Node                           │
│  PIR ──┐                                                    │
│  ACS712─┼──► Feature Vector (every 5 min) ──POST /ingest──► │
│  MAX30102┘                                                  │
│                                                             │
│  ◄── POST /status (AI score + severity feedback) ──────────┘
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  ingestion_server.py  │
                  │  (Flask, port 7777)   │
                  │                       │
                  │  1. Validate          │
                  │  2. Score inline (ML) │
                  │  3. Classify label    │
                  └──────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ThingsBoard Cloud              POST /status → ESP32
    (telemetry + ML scores,        (LED / buzzer actuation)
     alerts, dashboards)
              │
              ▼
       dashboard.py
       (port 5001, local
        real-time web UI)
```

**Key design principles:**
- **No local database** — all persistent storage is in ThingsBoard. The backend keeps a 100-vector in-memory ring buffer for the dashboard.
- **Inline scoring** — every ingest request is scored immediately; no separate scorer process.
- **Single Flask process** — the entire backend is one threaded server.
- **Bidirectional feedback** — the backend sends AI results back to the ESP32 for physical alerts.

---

## 3. ESP32 Firmware — How It Works

### 3.1 Sensors & Pins

| Sensor | Pin | Purpose |
|---|---|---|
| PIR (passive infrared) | GPIO 14 | Motion detection |
| ACS712 (current) | GPIO 34 (ADC) | Appliance current draw |
| MAX30102 (pulse-ox) | I²C (Wire) | Heart rate + SpO₂ |
| MAX7219 LED matrix | SPI (23/18/5) | Visual alert indicator |
| Buzzer | GPIO 27 | Audio alert |

### 3.2 Windowed Sampling (5-Minute Windows)

The firmware runs a **5-minute rolling window** (`WINDOW_MS = 300,000 ms`). Within each window:

1. **PIR** — every GPIO edge (LOW→HIGH) is counted. Active time (`pirActiveMs`) is accumulated.
2. **ACS712** — sampled every **500 ms** (up to 600 samples/window). Each sample is read as RMS current:
   ```
   I_rms = sqrt( mean( (V_mv - V_zero)² ) ) / sensitivity
   ```
   An appliance is considered `ON` if `I_rms > 0.05 A`.
3. **MAX30102** — polled continuously. When IR > 20,000 (finger detected), 100 samples are collected and passed to Maxim's `maxim_heart_rate_and_oxygen_saturation()` algorithm. Valid readings (HR: 35–220 bpm, SpO₂: 70–100%) are accumulated (up to 10 readings/window).

### 3.3 Feature Computation (`computeAndTransmit`)

At the end of each window, the ESP32 computes and builds a `FeatureVector`:

```
pir_active_ratio  = pirActiveMs / windowDuration
time_since_motion = (now - lastPirTrigger) / 1000.0   [seconds]
current_mean/std  = mean/std of ACS712 current samples
appliance_on_flag = any sample above threshold
on_duration_s     = total appliance ON time
hr_mean/std       = mean/std of valid HR readings
spo2_mean/std     = mean/std of valid SpO₂ readings
hour_sin/cos      = sin/cos encoding of current hour (cyclical)

tachycardia_flag  = 1 if hr_mean > 110 bpm
bradycardia_flag  = 1 if hr_mean < 50 bpm (and > 0)
spo2_low_flag     = 1 if spo2_mean < 92%

motion_and_appliance      = 1 if (pir_ratio > 0.05 AND appliance_on)
motion_no_appliance       = 1 if (pir_ratio > 0.05 AND NOT appliance_on)
no_motion_abnormal_vitals = 1 if (no motion AND abnormal vitals AND vitals_valid)
```

**Local rule-based alerts** are triggered immediately on the device (before backend ML scores arrive):
- No motion > 3600 s during daytime (07:00–21:00) → CRITICAL:PROLONGED_INACTIVITY
- Appliance ON > 1800 s → WARNING:APPLIANCE_LONG
- Tachycardia → WARNING:TACHYCARDIA
- Bradycardia → WARNING:BRADYCARDIA
- Low SpO₂ → WARNING:LOW_SPO2

### 3.4 Transmission & Retry

The feature vector is serialized to JSON and POSTed to `POST /ingest`. If transmission fails, the vector is stored in a 5-slot retry queue and replayed on the next successful window. If the queue is full, the oldest entry is dropped.

### 3.5 FreeRTOS Alert Queue (Cross-Core Safety)

The ESP32 runs on dual cores:
- **Core 0** — AsyncWebServer (handles incoming `/status` POSTs from backend)
- **Core 1** — `loop()` (sensor polling, window management, actuation)

A **FreeRTOS queue** (`alertQueue`, depth 10) safely delivers `AlertMsg` structs from the async callback (Core 0) to `loop()` (Core 1), avoiding cross-core String race conditions.

---

## 4. Feature Vector

The complete JSON structure sent from ESP32 to backend:

| Field | Type | Description |
|---|---|---|
| `ts` | int | Timestamp (ms since boot) |
| `hour` | int | Hour of day (0–23) |
| `hour_sin` | float | sin(2π·hour/24) — cyclical time encoding |
| `hour_cos` | float | cos(2π·hour/24) — cyclical time encoding |
| `pir_count` | int | Number of motion trigger events in window |
| `pir_ratio` | float | Fraction of window with active motion [0–1] |
| `no_motion_s` | float | Seconds since last motion (−1 = never) |
| `cur_mean` | float | Mean current draw (Amperes) |
| `cur_std` | float | Std deviation of current |
| `app_on` | int | 1 if appliance was ON during window |
| `app_on_s` | float | Total appliance ON duration (seconds) |
| `app_switches` | int | Number of ON/OFF transitions |
| `hr_mean` | float | Mean heart rate (bpm) |
| `hr_std` | float | Std deviation of heart rate |
| `spo2_mean` | float | Mean SpO₂ (%) |
| `spo2_std` | float | Std deviation of SpO₂ |
| `tachy` | int | 1 if tachycardia (HR > 110) |
| `brady` | int | 1 if bradycardia (HR < 50) |
| `spo2_low` | int | 1 if SpO₂ < 92% |
| `vitals_valid` | int | 1 if MAX30102 produced valid readings |
| `mot_and_app` | int | 1 if motion AND appliance both active |
| `mot_no_app` | int | 1 if motion but no appliance |
| `nomot_abnvit` | int | 1 if no motion AND abnormal vitals |

---

## 5. Backend — Ingestion Server

`ingestion_server.py` is a Flask server (port 7777) with these endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | POST | Receive feature vector → score → push to ThingsBoard → feedback to ESP32 |
| `/health` | GET | Server health + feature flags |
| `/status` | GET | Last N scored vectors from in-memory ring buffer |
| `/upload` | POST | Replay a `patient_data.json` through the ML pipeline |
| `/patient` | GET | Return current patient metadata |
| `/esp32/set` | POST | Dynamically update the ESP32 IP target |
| `/esp32/ping` | GET | Reachability check to the ESP32 |
| `/esp32/silence` | POST | Send SILENCED command to stop ESP32 alert |

### Ingest Pipeline (per request)

```
1. JSON parse + validate (23 required fields, type-checked, hour 0–23)
2. Normalise: no_motion_s == -1  →  300.0
3. score_vector(data)  →  {a_motion, a_adl, a_vitals, a_global, anomaly_label}
4. Log with severity icon (🟢/🟡/🔴)
5. push_to_thingsboard(data, scores)   [non-blocking]
6. send_esp32_feedback(label, score)   [non-blocking]
7. Store in _recent deque (maxlen=100)
8. Return JSON {status, anomaly_label, a_global, severity}
```

---

## 6. ML Pipeline & Algorithms

Three separate models cover three behavioural domains. All models are trained on **normal-routine data only** (unsupervised — no labelled anomaly examples required).

### 6.1 Motion Model — Isolation Forest

**Features:** `pir_count`, `pir_ratio`, `no_motion_s`, `hour_sin`, `hour_cos`

**How it works:**
- Builds an ensemble of 200 random isolation trees.
- Anomalies are isolated in **fewer splits** because they are rare and extreme.
- The raw score `∈ ≈ [-1, 0]` is shifted to `[0, 1]`:

```python
a_motion = clip(1.0 + model.score_samples(X_scaled), 0.0, 1.0)
# 0.0 = normal, 1.0 = maximally anomalous
```

`contamination=0.05`, `n_estimators=200`, trained on `StandardScaler`-normalized data.

### 6.2 ADL Model — Isolation Forest

**Features:** `cur_mean`, `cur_std`, `app_on`, `app_on_s`, `app_switches`, `hour_sin`, `hour_cos`

Same Isolation Forest algorithm applied to appliance/ADL patterns. Detects:
- Appliance left on for an unusually long time
- Unusual current draw at unexpected hours
- Abnormal on/off switching frequency

### 6.3 Vitals Model — Dense Autoencoder

**Features:** `hr_mean`, `hr_std`, `spo2_mean`, `spo2_std`

**Architecture (PyTorch):**
```
Input(4) → Linear(4→8) → ReLU → Linear(8→4) → ReLU → Linear(4→2)   [encoder]
           Linear(2→4) → ReLU → Linear(4→8) → ReLU → Linear(8→4)   [decoder]
```
The 2-dimensional bottleneck forces learning of the normal vitals manifold.

**Training:** 85/15 split, 100 epochs, Adam (lr=1e-3), StepLR scheduler. Best val-loss weights kept.

**Scoring:**
```python
err      = mean((X - model(X))²)         # reconstruction MSE
a_vitals = clip(err / vitals_crit, 0, 1) # vitals_crit = 99th pct of training errors
```

**No-vitals fallback:** `vitals_valid == 0` → `a_vitals = 0.5` (neutral).  
**Data fallback:** < 20 valid vitals rows → uses Isolation Forest instead.

---

## 7. Anomaly Scoring & Fusion

### Weighted Linear Fusion

```
a_global = 0.30 × a_motion  +  0.35 × a_adl  +  0.35 × a_vitals
```

All scores are in **[0, 1]** (0 = normal, 1 = maximally anomalous).

### Severity Thresholds

| a_global | Severity |
|---|---|
| < 0.65 | 🟢 NORMAL |
| 0.65 – 0.79 | 🟡 WARNING |
| ≥ 0.80 | 🔴 CRITICAL |

---

## 8. Anomaly Classification Rules

```python
if a_global < 0.65:
    → "NORMAL"

elif a_global >= 0.80 and nomot_abnvit == 1:
    → "CRITICAL:PHYSIOLOGICAL_EVENT"
    # No motion + abnormal vitals = possible medical emergency

elif a_global >= 0.80 and no_motion_s > 3600 and not night:
    → "CRITICAL:PROLONGED_INACTIVITY"
    # Daytime, no movement for over 1 hour

elif a_global >= 0.80:
    → "CRITICAL:GENERAL"

elif a_motion > 0.65 and hour in {22..5} and mot_no_app == 1:
    → "WARNING:NIGHT_WANDERING"
    # Moving at night with no appliance = possible disorientation

elif a_vitals > 0.65 and (tachy or brady or spo2_low):
    → "WARNING:PHYSIOLOGICAL"

elif a_adl > 0.65 and not app_on:
    → "WARNING:ROUTINE_ANOMALY"

else:
    → "WARNING:GENERAL"
```

Night = hours 22–05. Daytime inactivity check = hours 07–21.

---

## 9. Feedback Loop (ESP32 ↔ Backend)

After every scored vector the backend POSTs to the ESP32:

```json
POST http://<ESP32_IP>:8080/status
{ "status": "CRITICAL", "label": "CRITICAL:PROLONGED_INACTIVITY", "a_global": 0.91 }
```

| Status | ESP32 Action |
|---|---|
| `CRITICAL` | Blink LED + 1000 Hz buzzer for 40 s |
| `WARNING` | Blink LED + 800 Hz buzzer for 40 s |
| `SILENCED` | Immediately stop all indicators |
| `OK` | No action |

The dashboard's **SILENCE ALERT** button sends `POST /esp32/silence`, relaying `SILENCED` to the ESP32.

---

## 10. ThingsBoard Integration

Telemetry keys pushed per vector:

| Key | Description |
|---|---|
| `a_global` | Composite anomaly score [0, 1] |
| `anomaly_label` | Human-readable label |
| `severity` | `NORMAL` / `WARNING` / `CRITICAL` |
| `hr_mean`, `spo2_mean` | Latest vitals |
| `pir_ratio` | Motion activity ratio |
| `app_on`, `app_on_s` | Appliance state |
| `a_motion`, `a_adl`, `a_vitals` | Individual sub-scores |

Config via `.env.thingsboard`:
```bash
TB_ACCESS_TOKEN="your_device_access_token"
TB_HOST="demo.thingsboard.io"
ESP32_URL="192.168.x.x"
```

---

## 11. Dashboard (`dashboard.py`, port 5001)

**Modes:**
- **LIVE** — polls `GET /status` every 5 s; displays real-time scored vectors.
- **REPLAY** — drag-and-drop `patient_data.json` to replay historic readings via `POST /upload`.

**Displayed information:**
- Sensor state: PIR dot (pulsing), HR bars, SpO₂ arc, current bar, no-motion timer, LED matrix simulation, buzzer state
- AI sub-score bars: A_motion, A_adl, A_vitals with colour coding
- Stats: total readings, estimated hours, latest HR/SpO₂/A_global, alert count
- Charts: anomaly timeline (with threshold reference lines), sub-scores, HR, PIR ratio, SpO₂, appliance ON duration
- Anomaly event log: last 20 non-NORMAL events
- Patient profile panel (shown when JSON has patient metadata)
- ESP32 panel: IP config, PING, SILENCE ALERT

---

## 12. Simulator (`simulator.py`)

Replaces the physical ESP32 by POSTing synthetic feature vectors to `/ingest`.

| Scenario | Expected Output |
|---|---|
| `normal_day` | NORMAL |
| `normal_night` | NORMAL |
| `prolonged_inactivity` | CRITICAL:PROLONGED_INACTIVITY |
| `night_wandering` | WARNING:NIGHT_WANDERING |
| `tachycardia_event` | WARNING:PHYSIOLOGICAL |
| `low_spo2_event` | CRITICAL:PHYSIOLOGICAL_EVENT |

**Bulk seeding** (`--bulk-days N`): generates 288 vectors/day following a realistic hourly activity cycle (sleep, morning prep, active morning, lunch, afternoon, evening, winding down) with 7% random anomaly injection.

---

## 13. Model Training (`train_models.py`)

Requires ≥ 100 feature vectors (~8.3 hours of data).

**Steps:**
1. Load data from `dementia_monitor.db`
2. Clean: drop nulls, replace `no_motion_s == -1 → 300`
3. Train Motion IsolationForest
4. Train ADL IsolationForest
5. Train VitalsAutoencoder (or IsolationForest fallback if < 20 valid vitals rows)
6. Compute warn/crit thresholds at 95th/99th percentiles
7. Save all models + `training_meta.json`
8. Verify: score a dummy normal sample

**Output files (`models/`):**

| File | Contents |
|---|---|
| `motion_isoforest.joblib` | Motion anomaly model |
| `motion_scaler.joblib` | StandardScaler for motion features |
| `adl_isoforest.joblib` | ADL anomaly model |
| `adl_scaler.joblib` | StandardScaler for ADL features |
| `vitals_autoencoder.pt` | Autoencoder weights |
| `vitals_scaler.joblib` | StandardScaler for vitals |
| `training_meta.json` | Thresholds, fusion weights, feature lists |

---

## 14. Alert Severity Summary

| Label | Condition | Severity | ESP32 Response |
|---|---|---|---|
| `NORMAL` | a_global < 0.65 | 🟢 | No alert |
| `WARNING:NIGHT_WANDERING` | Night motion, no appliance, a_motion > 0.65 | 🟡 | 800 Hz, 40 s |
| `WARNING:PHYSIOLOGICAL` | Abnormal vitals, a_vitals > 0.65 | 🟡 | 800 Hz, 40 s |
| `WARNING:ROUTINE_ANOMALY` | ADL anomaly, no appliance | 🟡 | 800 Hz, 40 s |
| `WARNING:GENERAL` | a_global 0.65–0.79 | 🟡 | 800 Hz, 40 s |
| `WARNING:APPLIANCE_LONG` | Appliance on > 30 min (local rule) | 🟡 | 900 Hz, 40 s |
| `CRITICAL:PROLONGED_INACTIVITY` | Daytime, no motion > 1 h | 🔴 | 1000 Hz, 40 s |
| `CRITICAL:PHYSIOLOGICAL_EVENT` | No motion + abnormal vitals | 🔴 | 1000 Hz, 40 s |
| `CRITICAL:GENERAL` | a_global ≥ 0.80 | 🔴 | 1000 Hz, 40 s |

---

## 15. Data Flow Diagram

```
ESP32 (every 5 min)
  │
  ├── PIR GPIO ────────► pir_count, pir_ratio, time_since_last_motion
  ├── ACS712 ADC ───────► cur_mean, cur_std, app_on, app_on_s
  ├── MAX30102 I²C ─────► hr_mean, hr_std, spo2_mean, spo2_std
  └── NTP ──────────────► hour → hour_sin, hour_cos
                              │
                    Compute cross-modal flags
                    Apply local rule alerts
                    Serialize → JSON
                              │
                    POST /ingest (port 7777)
                              │
                    ┌─────────▼──────────┐
                    │   Validate (23 fields)  │
                    │   score_vector()        │
                    │                         │
                    │  IsoForest → a_motion   │
                    │  IsoForest → a_adl      │
                    │  Autoencoder → a_vitals │
                    │                         │
                    │  a_global = weighted sum │
                    │  _interpret() → label    │
                    └──┬──────────────────┬───┘
                       │                  │
                       ▼                  ▼
             ThingsBoard             POST /status
             (cloud storage,         → ESP32:8080
              dashboards)            (LED + buzzer)
                       │
                       ▼
               dashboard.py (port 5001)
               (real-time charts + alert log)
```

---

*Axialert v3 — Phase 3 (Feedback Loop) | Last updated: 2026-04-27*
