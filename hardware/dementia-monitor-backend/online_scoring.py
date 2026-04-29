import sqlite3
import json
import time
import os
import numpy as np
import joblib
import requests
import torch
import torch.nn as nn
from datetime import datetime

_HERE          = os.path.dirname(os.path.abspath(__file__))  # FIX #24
DB_PATH        = os.path.join(_HERE, "dementia_monitor.db")
MODELS_DIR     = os.path.join(_HERE, "models")
SCORE_INTERVAL_S = 30          # Score new vectors every 30 seconds
ESP32_URL      = "172.24.130.139"   # FIX #10: no http:// prefix here
ESP32_PORT     = 8080
SEND_TO_ESP32  = True

# ─── Fusion weights (can be tuned) ──────────────────────────────────
# Loaded from training_meta.json — override here if needed
W_MOTION = 0.30
W_ADL    = 0.35
W_VITALS = 0.35

# ─── Alert thresholds ────────────────────────────────────────────────
THRESH_WARN = 0.65
THRESH_CRIT = 0.80

# ─── Feature groups (must match train_models.py) ─────────────────────
MOTION_FEATURES  = ["pir_count", "pir_ratio", "no_motion_s", "hour_sin", "hour_cos"]
ADL_FEATURES     = ["cur_mean", "cur_std", "app_on", "app_on_s", "app_switches", "hour_sin", "hour_cos"]
VITALS_FEATURES  = ["hr_mean", "hr_std", "spo2_mean", "spo2_std"]

# ─── Autoencoder definition (must match train_models.py) ─────────────
class VitalsAutoencoder(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, 4),         nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),         nn.ReLU(),
            nn.Linear(4, 8),         nn.ReLU(),
            nn.Linear(8, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── Model loader ────────────────────────────────────────────────────

class ModelBundle:
    """Loads and holds all models + metadata. Reloads if models dir changes."""

    def __init__(self):
        self.motion_model  = None
        self.motion_scaler = None
        self.adl_model     = None
        self.adl_scaler    = None
        self.vitals_model  = None
        self.vitals_scaler = None
        self.vitals_type   = None   # "autoencoder" or "isoforest"
        self.meta          = None
        self.loaded        = False
        self._load()

    def _load(self):
        meta_path = f"{MODELS_DIR}/training_meta.json"
        if not os.path.exists(meta_path):
            print("[SCORER] No trained models found. Run train_models.py first.")
            return

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.motion_model  = joblib.load(f"{MODELS_DIR}/motion_isoforest.joblib")
        self.motion_scaler = joblib.load(f"{MODELS_DIR}/motion_scaler.joblib")
        self.adl_model     = joblib.load(f"{MODELS_DIR}/adl_isoforest.joblib")
        self.adl_scaler    = joblib.load(f"{MODELS_DIR}/adl_scaler.joblib")
        self.vitals_scaler = joblib.load(f"{MODELS_DIR}/vitals_scaler.joblib")

        self.vitals_type = self.meta.get("vitals_model_type", "autoencoder")
        if self.vitals_type == "autoencoder":
            ae = VitalsAutoencoder(input_dim=len(VITALS_FEATURES))
            ae.load_state_dict(torch.load(f"{MODELS_DIR}/vitals_autoencoder.pt",
                                          map_location="cpu", weights_only=True))
            ae.eval()
            self.vitals_model = ae
        else:
            self.vitals_model = joblib.load(f"{MODELS_DIR}/vitals_isoforest.joblib")

        # Override fusion weights from meta if present
        global W_MOTION, W_ADL, W_VITALS
        fw = self.meta.get("fusion_weights", {})
        W_MOTION = fw.get("w_motion", W_MOTION)
        W_ADL    = fw.get("w_adl",    W_ADL)
        W_VITALS = fw.get("w_vitals", W_VITALS)

        self.loaded = True
        print(f"[SCORER] Models loaded. Trained on {self.meta['n_vectors']} vectors.")
        print(f"[SCORER] Thresholds — warn: {THRESH_WARN}, crit: {THRESH_CRIT}")
        print(f"[SCORER] Fusion weights — motion: {W_MOTION}, adl: {W_ADL}, vitals: {W_VITALS}")


# ─── Scoring functions ───────────────────────────────────────────────

def score_isoforest(model, scaler, X: np.ndarray) -> np.ndarray:
    """Normalised anomaly score [0, 1]. Higher = more anomalous."""
    X_scaled   = scaler.transform(X)
    raw        = model.score_samples(X_scaled)  # range ≈ [-1, 0]
    normalised = np.clip(1.0 + raw, 0.0, 1.0)   # shift to [0, 1]
    return normalised


def score_autoencoder(model, scaler, X: np.ndarray, crit_threshold: float) -> np.ndarray:
    """
    Reconstruction error normalised by the critical threshold.
    Returns score in [0, 1] — 1.0 means error >= critical threshold.
    """
    X_scaled = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        tensor = torch.tensor(X_scaled)
        recon  = model(tensor)
        errors = ((tensor - recon) ** 2).mean(dim=1).numpy()
    normalised = np.clip(errors / (crit_threshold + 1e-9), 0.0, 1.0)
    return normalised


def compute_scores(bundle: ModelBundle, row: dict):
    """
    Compute A_motion, A_adl, A_vitals, A_global for a single feature vector row.
    Returns dict of scores.
    """
    thresholds = bundle.meta["thresholds"]

    # ── Motion score ──
    X_motion = np.array([[row[f] for f in MOTION_FEATURES]])
    a_motion = float(score_isoforest(bundle.motion_model, bundle.motion_scaler, X_motion)[0])

    # ── ADL score ──
    X_adl = np.array([[row[f] for f in ADL_FEATURES]])
    a_adl = float(score_isoforest(bundle.adl_model, bundle.adl_scaler, X_adl)[0])

    # ── Vitals score ──
    if row["vitals_valid"] == 1:
        X_vitals = np.array([[row[f] for f in VITALS_FEATURES]])
        if bundle.vitals_type == "autoencoder":
            a_vitals = float(score_autoencoder(
                bundle.vitals_model, bundle.vitals_scaler, X_vitals,
                thresholds["vitals_crit"]
            )[0])
        else:
            a_vitals = float(score_isoforest(
                bundle.vitals_model, bundle.vitals_scaler, X_vitals
            )[0])
    else:
        # No vitals reading this window — use neutral mid score
        a_vitals = 0.5

    # ── Weighted fusion ──
    a_global = W_MOTION * a_motion + W_ADL * a_adl + W_VITALS * a_vitals

    return {
        "a_motion":  round(a_motion, 4),
        "a_adl":     round(a_adl,    4),
        "a_vitals":  round(a_vitals,  4),
        "a_global":  round(a_global,  4),
    }


# ─── Rule-based interpretation ───────────────────────────────────────

def interpret_anomaly(scores: dict, row: dict) -> str:
    """
    Maps score + cross-modal features to a human-readable anomaly label.
    Returns one of:
        NORMAL
        WARNING:NIGHT_WANDERING
        WARNING:ROUTINE_ANOMALY
        WARNING:PHYSIOLOGICAL
        WARNING:GENERAL
        CRITICAL:PHYSIOLOGICAL_EVENT
        CRITICAL:PROLONGED_INACTIVITY
        CRITICAL:GENERAL
    """
    a_global  = scores["a_global"]
    a_motion  = scores["a_motion"]
    a_adl     = scores["a_adl"]
    a_vitals  = scores["a_vitals"]

    pir_active      = row["pir_ratio"] > 0.05
    app_on          = row["app_on"] == 1
    abnormal_vitals = row["tachy"] or row["brady"] or row["spo2_low"]
    is_night        = row["hour"] >= 22 or row["hour"] <= 5
    no_motion_long  = row["no_motion_s"] > 3600

    # Normal
    if a_global < THRESH_WARN:
        return "NORMAL"

    # ── Critical rules ──

    # Possible collapse / health event: no motion + abnormal vitals
    if a_global >= THRESH_CRIT and row["nomot_abnvit"] == 1:
        return "CRITICAL:PHYSIOLOGICAL_EVENT"

    # No motion for very long during day
    if a_global >= THRESH_CRIT and no_motion_long and not is_night:
        return "CRITICAL:PROLONGED_INACTIVITY"

    if a_global >= THRESH_CRIT:
        return "CRITICAL:GENERAL"

    # ── Warning rules ──

    # Night wandering: motion active at night, no appliance
    if a_motion > THRESH_WARN and is_night and row["mot_no_app"] == 1:
        return "WARNING:NIGHT_WANDERING"

    # Physiological anomaly: vitals-driven
    if a_vitals > THRESH_WARN and abnormal_vitals:
        return "WARNING:PHYSIOLOGICAL"

    # Routine anomaly: ADL-driven (skipped meal, unusual appliance pattern)
    if a_adl > THRESH_WARN and not app_on:
        return "WARNING:ROUTINE_ANOMALY"

    return "WARNING:GENERAL"


# ─── Database helpers ────────────────────────────────────────────────

def get_unscored_vectors(conn, limit=50):
    """Fetch vectors that haven't been scored yet."""
    rows = conn.execute(
        """
        SELECT * FROM feature_vectors
        WHERE a_global IS NULL
        ORDER BY id ASC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def write_scores(conn, vector_id: int, scores: dict, label: str):
    conn.execute(
        """
        UPDATE feature_vectors
        SET a_motion = ?, a_adl = ?, a_vitals = ?, a_global = ?, anomaly_label = ?
        WHERE id = ?
        """,
        (scores["a_motion"], scores["a_adl"], scores["a_vitals"],
         scores["a_global"], label, vector_id)
    )


# ─── ESP32 feedback ──────────────────────────────────────────────────

def send_status_to_esp32(label: str, a_global: float):
    """
    POST anomaly status back to ESP32.
    ESP32 Phase 1 Step 3 doesn't have a receiver yet —
    this will be wired up when you add a /status endpoint to the ESP32.
    """
    if not SEND_TO_ESP32:
        return

    severity = "CRITICAL" if "CRITICAL" in label else \
               "WARNING"  if "WARNING"  in label else "OK"

    payload = {
        "status":   severity,
        "label":    label,
        "a_global": a_global,
    }
    try:
        resp = requests.post(
            f"http://{ESP32_URL}:{ESP32_PORT}/status",  # FIX #10: URL now valid
            json=payload, timeout=3
        )
        if resp.status_code == 200:
            print(f"[ESP32] Status sent: {severity}")
    except requests.exceptions.RequestException as e:
        print(f"[ESP32] Send failed: {e}")


# ─── Main scoring loop ───────────────────────────────────────────────

def run_scorer():
    print("=" * 55)
    print("  Dementia Monitor — Phase 2 Online Scorer")
    print("=" * 55)

    bundle = ModelBundle()
    if not bundle.loaded:
        print("[SCORER] Waiting for models... re-checking every 60s")
        while not bundle.loaded:
            time.sleep(60)
            bundle._load()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # FIX #4: enable WAL for concurrent access

    print(f"[SCORER] Running. Scoring new vectors every {SCORE_INTERVAL_S}s")
    print("[SCORER] Press Ctrl+C to stop.\n")

    scored_total = 0

    while True:
        try:
            rows = get_unscored_vectors(conn, limit=50)

            if rows:
                for row in rows:
                    # Replace -1 no_motion_s with 300 (same as training)
                    if row["no_motion_s"] == -1:
                        row["no_motion_s"] = 300

                    scores = compute_scores(bundle, row)
                    label  = interpret_anomaly(scores, row)

                    write_scores(conn, row["id"], scores, label)
                    conn.commit()
                    scored_total += 1

                    # Console output
                    severity = label.split(":")[0]
                    icon = "🔴" if severity == "CRITICAL" else \
                           "🟡" if severity == "WARNING"  else "🟢"
                    print(
                        f"{icon} #{row['id']:4d} | "
                        f"h={row['hour']:02d} | "
                        f"A_motion={scores['a_motion']:.3f} "
                        f"A_adl={scores['a_adl']:.3f} "
                        f"A_vitals={scores['a_vitals']:.3f} "
                        f"A_global={scores['a_global']:.3f} | "
                        f"{label}"
                    )

                    # Send critical alerts to ESP32
                    if severity in ("CRITICAL", "WARNING"):
                        send_status_to_esp32(label, scores["a_global"])

                    time.sleep(0.05)  # FIX #11: yield CPU between rows

                print(f"[SCORER] Scored {len(rows)} vectors "
                      f"(total: {scored_total})\n")

            else:
                # Nothing new — wait quietly
                time.sleep(SCORE_INTERVAL_S)

        except KeyboardInterrupt:
            print(f"\n[SCORER] Stopped. Total scored: {scored_total}")
            break
        except Exception as e:
            print(f"[SCORER] Error: {e}")
            time.sleep(10)

    conn.close()


if __name__ == "__main__":
    run_scorer()