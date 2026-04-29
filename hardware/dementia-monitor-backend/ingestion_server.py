"""
ingestion_server.py  —  Axialert ThingsBoard-first ingestion server
====================================================================
Architecture (no local database):

    ESP32  ──POST /ingest──▶  this server
                                  │
                           score inline (ML)
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
               ThingsBoard             POST /status back
               (telemetry +            to ESP32 (LED /
                ML scores)              buzzer feedback)

SETUP
-----
1. Copy .env.thingsboard.example to .env.thingsboard and fill in your token:
       TB_ACCESS_TOKEN="your_device_access_token"
       TB_HOST="demo.thingsboard.io"   # or your self-hosted instance

2. Source the env file before starting:
       source .env.thingsboard

3. Run:
       python ingestion_server.py

ENDPOINTS
---------
  POST /ingest       Receive feature vector from ESP32
  GET  /health       Server health check
  GET  /status       Latest scored vector (in-memory, last 100)
"""

from flask import Flask, request, jsonify
import os, json, time, logging
from datetime import datetime, timezone
from collections import deque

import math as _math
import numpy as np
import joblib
import torch
import torch.nn as nn
import requests as _requests

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("axialert")

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")

TB_ACCESS_TOKEN = os.getenv("TB_ACCESS_TOKEN", "")
TB_HOST         = os.getenv("TB_HOST",         "demo.thingsboard.io")
TB_USE_HTTPS    = os.getenv("TB_USE_HTTPS",    "true").lower() in {"1","true","yes","on"}
TB_TIMEOUT_S    = int(os.getenv("TB_HTTP_TIMEOUT_SEC", "5"))
TB_ENABLED      = bool(TB_ACCESS_TOKEN)

_tb_scheme      = "https" if TB_USE_HTTPS else "http"
TB_TELEMETRY_URL = f"{_tb_scheme}://{TB_HOST}/api/v1/{TB_ACCESS_TOKEN}/telemetry"

ESP32_URL       = os.getenv("ESP32_URL",  "")   # e.g. "172.24.130.139"
ESP32_PORT      = int(os.getenv("ESP32_PORT", "8080"))
SEND_TO_ESP32   = bool(ESP32_URL)

# Mutable at runtime via POST /esp32/set
_esp32_ip: str  = ESP32_URL
_esp32_port: int = ESP32_PORT

# ─── Thresholds ───────────────────────────────────────────────────────────────
THRESH_WARN = 0.65
THRESH_CRIT = 0.80

MOTION_FEATURES = ["pir_count","pir_ratio","no_motion_s","hour_sin","hour_cos"]
ADL_FEATURES    = ["cur_mean","cur_std","app_on","app_on_s","app_switches","hour_sin","hour_cos"]
VITALS_FEATURES = ["hr_mean","hr_std","spo2_mean","spo2_std"]

# ─── Autoencoder (must match train_models.py) ─────────────────────────────────
class VitalsAutoencoder(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU(),nn.Linear(4,2))
        self.decoder = nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,8),nn.ReLU(),nn.Linear(8,input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

# ─── Model bundle (loaded once at startup) ────────────────────────────────────
class Models:
    loaded    = False
    meta      = None
    motion_m  = motion_sc = None
    adl_m     = adl_sc    = None
    vitals_m  = vitals_sc = None
    vitals_type = "autoencoder"

_models = Models()

def load_models():
    meta_path = os.path.join(MODELS_DIR, "training_meta.json")
    if not os.path.exists(meta_path):
        log.warning("No trained models found at %s — scoring disabled. "
                    "Run train_models.py first.", MODELS_DIR)
        return False
    with open(meta_path) as f:
        _models.meta = json.load(f)
    _models.motion_m  = joblib.load(os.path.join(MODELS_DIR, "motion_isoforest.joblib"))
    _models.motion_sc = joblib.load(os.path.join(MODELS_DIR, "motion_scaler.joblib"))
    _models.adl_m     = joblib.load(os.path.join(MODELS_DIR, "adl_isoforest.joblib"))
    _models.adl_sc    = joblib.load(os.path.join(MODELS_DIR, "adl_scaler.joblib"))
    _models.vitals_sc = joblib.load(os.path.join(MODELS_DIR, "vitals_scaler.joblib"))
    _models.vitals_type = _models.meta.get("vitals_model_type", "autoencoder")
    if _models.vitals_type == "autoencoder":
        ae = VitalsAutoencoder(input_dim=len(VITALS_FEATURES))
        ae.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "vitals_autoencoder.pt"),
            map_location="cpu", weights_only=True))
        ae.eval()
        _models.vitals_m = ae
    else:
        _models.vitals_m = joblib.load(os.path.join(MODELS_DIR, "vitals_isoforest.joblib"))
    fw = _models.meta.get("fusion_weights", {})
    _models.w_motion = fw.get("w_motion", 0.30)
    _models.w_adl    = fw.get("w_adl",    0.35)
    _models.w_vitals = fw.get("w_vitals",  0.35)
    _models.loaded = True
    log.info("Models loaded — trained on %d vectors", _models.meta["n_vectors"])
    return True

# ─── Scoring ──────────────────────────────────────────────────────────────────
def _iso_score(model, scaler, X):
    return float(np.clip(1.0 + model.score_samples(scaler.transform(X)), 0.0, 1.0)[0])

def _ae_score(model, scaler, X, crit):
    X_s = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        t = torch.tensor(X_s)
        err = ((t - model(t)) ** 2).mean(dim=1).numpy()[0]
    return float(np.clip(err / (crit + 1e-9), 0.0, 1.0))

def score_vector(data: dict) -> dict:
    if not _models.loaded:
        return {"a_motion": None, "a_adl": None, "a_vitals": None,
                "a_global": None, "anomaly_label": "UNSCORED"}

    Xm = np.array([[data[f] for f in MOTION_FEATURES]])
    Xa = np.array([[data[f] for f in ADL_FEATURES]])
    a_motion = _iso_score(_models.motion_m, _models.motion_sc, Xm)
    a_adl    = _iso_score(_models.adl_m,    _models.adl_sc,    Xa)

    if data.get("vitals_valid") == 1:
        Xv = np.array([[data[f] for f in VITALS_FEATURES]])
        if _models.vitals_type == "autoencoder":
            a_vitals = _ae_score(_models.vitals_m, _models.vitals_sc, Xv,
                                 _models.meta["thresholds"]["vitals_crit"])
        else:
            a_vitals = _iso_score(_models.vitals_m, _models.vitals_sc, Xv)
    else:
        a_vitals = 0.5   # neutral when no vitals

    a_global = (_models.w_motion * a_motion +
                _models.w_adl    * a_adl    +
                _models.w_vitals * a_vitals)

    scores = {
        "a_motion": round(a_motion, 4),
        "a_adl":    round(a_adl,   4),
        "a_vitals": round(a_vitals, 4),
        "a_global": round(a_global, 4),
    }
    scores["anomaly_label"] = _interpret(scores, data)
    return scores

def _interpret(scores: dict, data: dict) -> str:
    g = scores["a_global"]
    if g < THRESH_WARN:
        return "NORMAL"
    pir_active     = data["pir_ratio"] > 0.05
    app_on         = data["app_on"] == 1
    abnorm_vitals  = data["tachy"] or data["brady"] or data["spo2_low"]
    is_night       = data["hour"] >= 22 or data["hour"] <= 5
    no_motion_long = data["no_motion_s"] > 3600
    if g >= THRESH_CRIT and data.get("nomot_abnvit") == 1:
        return "CRITICAL:PHYSIOLOGICAL_EVENT"
    if g >= THRESH_CRIT and no_motion_long and not is_night:
        return "CRITICAL:PROLONGED_INACTIVITY"
    if g >= THRESH_CRIT:
        return "CRITICAL:GENERAL"
    if scores["a_motion"] > THRESH_WARN and is_night and data.get("mot_no_app") == 1:
        return "WARNING:NIGHT_WANDERING"
    if scores["a_vitals"] > THRESH_WARN and abnorm_vitals:
        return "WARNING:PHYSIOLOGICAL"
    if scores["a_adl"] > THRESH_WARN and not app_on:
        return "WARNING:ROUTINE_ANOMALY"
    return "WARNING:GENERAL"

# ─── ThingsBoard push ─────────────────────────────────────────────────────────
def push_to_thingsboard(data: dict, scores: dict):
    if not TB_ENABLED:
        return
    payload = {
        # Raw telemetry
        "hour":         data["hour"],
        "pir_count":    data["pir_count"],
        "pir_ratio":    data["pir_ratio"],
        "no_motion_s":  data["no_motion_s"],
        "cur_mean":     data["cur_mean"],
        "app_on":       data["app_on"],
        "app_on_s":     data["app_on_s"],
        "hr_mean":      data["hr_mean"],
        "spo2_mean":    data["spo2_mean"],
        "vitals_valid": data["vitals_valid"],
        "tachy":        data["tachy"],
        "brady":        data["brady"],
        "spo2_low":     data["spo2_low"],
        # ML scores
        "a_motion":     scores.get("a_motion"),
        "a_adl":        scores.get("a_adl"),
        "a_vitals":     scores.get("a_vitals"),
        "a_global":     scores.get("a_global"),
        "anomaly_label":scores.get("anomaly_label", "UNKNOWN"),
        # Severity for easy ThingsBoard widget colouring
        "severity":     scores.get("anomaly_label","").split(":")[0] or "NORMAL",
    }
    try:
        r = _requests.post(TB_TELEMETRY_URL, json=payload, timeout=TB_TIMEOUT_S)
        if r.status_code == 200:
            log.debug("[TB] pushed OK")
        else:
            log.warning("[TB] push failed %d: %s", r.status_code, r.text[:80])
    except _requests.exceptions.RequestException as e:
        log.warning("[TB] push error: %s", e)

# ─── ESP32 feedback ───────────────────────────────────────────────────────────
def send_esp32_feedback(label: str, a_global: float):
    if not _esp32_ip:
        return
    sev = ("CRITICAL" if "CRITICAL" in label else
           "WARNING"  if "WARNING"  in label else "OK")
    try:
        r = _requests.post(
            f"http://{_esp32_ip}:{_esp32_port}/status",
            json={"status": sev, "label": label, "a_global": a_global},
            timeout=3,
        )
        if r.status_code == 200:
            log.info("[ESP32] feedback sent: %s", sev)
        else:
            log.warning("[ESP32] feedback HTTP %d", r.status_code)
    except _requests.exceptions.RequestException as e:
        log.warning("[ESP32] feedback failed: %s", e)


@app.route("/esp32/set", methods=["POST"])
def esp32_set():
    global _esp32_ip, _esp32_port
    body = request.get_json(force=True) or {}
    ip   = body.get("ip",   "").strip()
    port = int(body.get("port", _esp32_port))
    if not ip:
        return jsonify({"error": "ip required"}), 400
    _esp32_ip   = ip
    _esp32_port = port
    log.info("[ESP32] target set to %s:%d", _esp32_ip, _esp32_port)
    return jsonify({"ok": True, "ip": _esp32_ip, "port": _esp32_port})


@app.route("/esp32/silence", methods=["POST"])
def esp32_silence():
    if not _esp32_ip:
        return jsonify({"ok": False, "error": "No ESP32 configured"}), 400
    try:
        r = _requests.post(
            f"http://{_esp32_ip}:{_esp32_port}/status",
            json={"status": "SILENCED", "label": "SILENCED", "a_global": 0.0},
            timeout=3,
        )
        if r.status_code == 200:
            log.info("[ESP32] alert silenced by user")
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": f"HTTP {r.status_code}"}), 502
    except _requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/esp32/ping", methods=["GET"])
def esp32_ping():
    if not _esp32_ip:
        return jsonify({"reachable": False, "reason": "no IP configured"})
    try:
        r = _requests.get(f"http://{_esp32_ip}:{_esp32_port}/health", timeout=2)
        return jsonify({"reachable": r.status_code == 200,
                        "ip": _esp32_ip, "port": _esp32_port,
                        "response": r.json() if r.ok else r.text[:80]})
    except _requests.exceptions.ConnectTimeout:
        return jsonify({"reachable": False, "ip": _esp32_ip, "port": _esp32_port,
                        "reason": f"Timed out — is {_esp32_ip} on the same network?"})
    except _requests.exceptions.ConnectionError:
        return jsonify({"reachable": False, "ip": _esp32_ip, "port": _esp32_port,
                        "reason": f"Connection refused — check ESP32 is powered and firmware is running"})
    except _requests.exceptions.RequestException as e:
        return jsonify({"reachable": False, "ip": _esp32_ip, "port": _esp32_port,
                        "reason": type(e).__name__})

# ─── In-memory ring buffer (last 100 scored vectors for /status) ───────────────
_recent: deque = deque(maxlen=100)

# ─── Validation ───────────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "ts":int,"hour":int,"hour_sin":float,"hour_cos":float,
    "pir_count":int,"pir_ratio":float,"no_motion_s":float,
    "cur_mean":float,"cur_std":float,"app_on":int,"app_on_s":float,"app_switches":int,
    "hr_mean":float,"hr_std":float,"spo2_mean":float,"spo2_std":float,
    "tachy":int,"brady":int,"spo2_low":int,"vitals_valid":int,
    "mot_and_app":int,"mot_no_app":int,"nomot_abnvit":int,
}

def validate(data: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    for f, t in REQUIRED_FIELDS.items():
        if not isinstance(data[f], (int, float)):
            return False, f"Field '{f}' has wrong type: {type(data[f]).__name__}"
    if not 0 <= data["hour"] <= 23:
        return False, f"hour out of range: {data['hour']}"
    return True, None

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    # 1. Parse
    try:
        data = request.get_json(force=True)
        if data is None:
            raise ValueError("Empty body")
    except Exception as e:
        return jsonify({"status":"error","message":f"JSON parse error: {e}"}), 400

    # 2. Validate
    ok, err = validate(data)
    if not ok:
        log.warning("[INGEST] validation failed: %s", err)
        return jsonify({"status":"error","message":err}), 422

    # 3. Normalise sentinel
    if data.get("no_motion_s") == -1:
        data["no_motion_s"] = 300.0

    # 4. Score inline — no database round-trip
    t0     = time.monotonic()
    scores = score_vector(data)
    label  = scores["anomaly_label"]
    dt_ms  = round((time.monotonic() - t0) * 1000, 1)

    # 5. Log
    sev  = label.split(":")[0]
    icon = "🔴" if sev == "CRITICAL" else "🟡" if sev == "WARNING" else "🟢"
    log.info("%s  h=%02d PIR=%.2f HR=%.0f SpO2=%.0f  A_global=%.3f  %s  (%sms)",
             icon, data["hour"], data["pir_ratio"], data["hr_mean"],
             data["spo2_mean"], scores.get("a_global") or 0, label, dt_ms)

    # 6. Push to ThingsBoard
    push_to_thingsboard(data, scores)

    # 7. Feedback to ESP32 (non-blocking — errors are logged, not fatal)
    send_esp32_feedback(label, scores.get("a_global") or 0.0)

    # 8. Store in memory ring buffer
    _recent.appendleft({**data, **scores,
                        "source": "live",
                        "received_at": datetime.now(timezone.utc).isoformat()})

    return jsonify({
        "status":        "ok",
        "anomaly_label": label,
        "a_global":      scores.get("a_global"),
        "severity":      sev,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "ok",
        "models_loaded":     _models.loaded,
        "thingsboard":       TB_ENABLED,
        "esp32_configured":  bool(_esp32_ip),
        "esp32_ip":          _esp32_ip or None,
        "esp32_port":        _esp32_port,
        "vectors_in_memory": len(_recent),
        "time":              datetime.now(timezone.utc).isoformat(),
    })


@app.route("/status", methods=["GET"])
def status():
    """Return the last N scored vectors from the in-memory buffer."""
    n = min(request.args.get("n", 20, type=int), 100)
    return jsonify(list(_recent)[:n])


# ─── Patient JSON upload (drag-and-drop) ──────────────────────────────────────
_patient_info: dict = {}

def _normalize_reading(r: dict) -> dict:
    hour = int(r.get("hour", 0))
    hs   = round(_math.sin(2 * _math.pi * hour / 24), 4)
    hc   = round(_math.cos(2 * _math.pi * hour / 24), 4)
    ts   = r.get("ts", 0)
    if isinstance(ts, str):
        try:
            from datetime import datetime as _dt
            ts = int(_dt.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            ts = int(time.time() * 1000)
    pir_ratio    = float(r.get("pir_ratio",   0.0))
    app_on       = int(r.get("app_on",        0))
    tachy        = int(r.get("tachy",         0))
    brady        = int(r.get("brady",         0))
    spo2_low     = int(r.get("spo2_low",      0))
    vitals_valid = int(r.get("vitals_valid",  0))
    pir_active   = pir_ratio > 0.05
    app_on_b     = app_on == 1
    abnorm_v     = bool(tachy or brady or spo2_low)
    return {
        "ts":           ts,
        "hour":         hour,
        "hour_sin":     hs,
        "hour_cos":     hc,
        "pir_count":    int(r.get("pir_count",    0)),
        "pir_ratio":    pir_ratio,
        "no_motion_s":  float(r.get("no_motion_s", 300.0)),
        "cur_mean":     float(r.get("cur_mean",    0.0)),
        "cur_std":      float(r.get("cur_std",     0.0)),
        "app_on":       app_on,
        "app_on_s":     float(r.get("app_on_s",   0.0)),
        "app_switches": int(r.get("app_switches",  0)),
        "hr_mean":      float(r.get("hr_mean",     0.0)),
        "hr_std":       float(r.get("hr_std",      0.0)),
        "spo2_mean":    float(r.get("spo2_mean",   0.0)),
        "spo2_std":     float(r.get("spo2_std",    0.0)),
        "tachy":        tachy,
        "brady":        brady,
        "spo2_low":     spo2_low,
        "vitals_valid": vitals_valid,
        "mot_and_app":  1 if (pir_active and app_on_b) else 0,
        "mot_no_app":   1 if (pir_active and not app_on_b) else 0,
        "nomot_abnvit": 1 if (not pir_active and abnorm_v and vitals_valid) else 0,
    }


@app.route("/upload", methods=["POST"])
def upload():
    global _patient_info
    body = request.get_json(force=True)
    if body is None:
        return jsonify({"error": "JSON parse error"}), 400
    patient  = body.get("patient", {})
    readings = body.get("readings", [])
    if not readings:
        return jsonify({"error": "No readings in file"}), 422

    _patient_info = patient
    results, skipped = [], 0
    for r in readings:
        try:
            vector = _normalize_reading(r)
            if vector.get("no_motion_s") == -1:
                vector["no_motion_s"] = 300.0
            scores = score_vector(vector)
            label  = scores["anomaly_label"]
            _recent.appendleft({
                **vector, **scores,
                "source": "replay",
                "received_at": datetime.now(timezone.utc).isoformat(),
            })
            send_esp32_feedback(label, scores.get("a_global") or 0.0)
            results.append({
                "ts":            vector["ts"],
                "hour":          vector["hour"],
                "a_global":      scores.get("a_global"),
                "anomaly_label": label,
            })
        except Exception as e:
            log.warning("[UPLOAD] skipping reading: %s", e)
            skipped += 1

    log.info("[UPLOAD] %d/%d readings processed for %s",
             len(results), len(readings), patient.get("name", "unknown"))
    return jsonify({"patient": patient, "processed": len(results),
                    "skipped": skipped, "results": results}), 200


@app.route("/patient", methods=["GET"])
def patient():
    return jsonify(_patient_info)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Axialert — ThingsBoard-first ingestion server")
    print("=" * 60)
    print(f"  ThingsBoard : {'ENABLED  → ' + TB_HOST if TB_ENABLED else 'DISABLED (set TB_ACCESS_TOKEN)'}")
    print(f"  ESP32 URL   : {'http://' + ESP32_URL + ':' + str(ESP32_PORT) if SEND_TO_ESP32 else 'not set (set ESP32_URL)'}")
    print(f"  Models      : {MODELS_DIR}")
    print()

    load_models()

    print()
    print("  Endpoint    : POST http://0.0.0.0:7777/ingest")
    print("  Health      : GET  http://0.0.0.0:7777/health")
    print("  Recent      : GET  http://0.0.0.0:7777/status?n=20")
    print()
    app.run(host="0.0.0.0", port=7777, debug=False, threaded=True)