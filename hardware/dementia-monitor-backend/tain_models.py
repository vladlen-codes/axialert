"""
train_models.py
Dementia Monitoring — Phase 2, Step 2
Model training: Isolation Forest (motion + ADL) + Autoencoder (vitals)

Run this after collecting 3-5 days of normal-routine data.

SETUP:
    pip install scikit-learn torch pandas numpy joblib

RUN:
    python train_models.py

OUTPUT:
    models/
        motion_isoforest.joblib   — motion anomaly model
        adl_isoforest.joblib      — appliance/ADL anomaly model
        vitals_autoencoder.pt     — vitals autoencoder weights
        vitals_scaler.joblib      — scaler for vitals features
        motion_scaler.joblib      — scaler for motion features
        adl_scaler.joblib         — scaler for ADL features
        training_meta.json        — thresholds and training stats
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DB_PATH     = "dementia_monitor.db"
MODELS_DIR  = "models"
MIN_VECTORS = 100   # Minimum rows needed before training (≈ 8 hours at 5-min windows)

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Feature groups ──────────────────────────────────────────────────

MOTION_FEATURES = [
    "pir_count", "pir_ratio", "no_motion_s",
    "hour_sin",  "hour_cos",
]

ADL_FEATURES = [
    "cur_mean", "cur_std", "app_on", "app_on_s", "app_switches",
    "hour_sin", "hour_cos",
]

VITALS_FEATURES = [
    "hr_mean", "hr_std", "spo2_mean", "spo2_std",
]

# ─── Load data ───────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run ingestion_server.py first.")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM feature_vectors ORDER BY id ASC", conn
    )
    conn.close()

    print(f"[DATA] Loaded {len(df)} vectors from {DB_PATH}")

    if len(df) < MIN_VECTORS:
        raise ValueError(
            f"Only {len(df)} vectors available. Need at least {MIN_VECTORS} "
            f"(≈ {MIN_VECTORS * 5 / 60:.1f} hours). Keep collecting data."
        )

    # Fill no_motion_s = -1 (never triggered) with window length as safe default
    df["no_motion_s"] = df["no_motion_s"].replace(-1, 300)

    # Drop rows where any required feature is null
    all_features = list(set(MOTION_FEATURES + ADL_FEATURES + VITALS_FEATURES))
    before = len(df)
    df = df.dropna(subset=all_features)
    if len(df) < before:
        print(f"[DATA] Dropped {before - len(df)} rows with null features")

    print(f"[DATA] Using {len(df)} clean vectors spanning "
          f"{df['received_at'].iloc[0]} → {df['received_at'].iloc[-1]}")

    return df

# ─── Isolation Forest ────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray, name: str, contamination=0.05):
    """
    Train an Isolation Forest on feature matrix X.
    contamination = estimated fraction of anomalies in training data.
    Set low (0.01-0.05) since training data should be mostly normal.
    Returns fitted model and scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Compute score distribution on training data
    raw_scores = model.score_samples(X_scaled)
    # Normalise to [0, 1] — higher = more anomalous
    scores_norm = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    print(f"[{name}] IsolationForest trained on {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[{name}] Score range: {scores_norm.min():.4f} – {scores_norm.max():.4f} "
          f"(mean: {scores_norm.mean():.4f})")

    # Threshold: 95th percentile of training scores as Warning boundary
    threshold_warn  = float(np.percentile(scores_norm, 95))
    threshold_crit  = float(np.percentile(scores_norm, 99))
    print(f"[{name}] Warning threshold:  {threshold_warn:.4f}")
    print(f"[{name}] Critical threshold: {threshold_crit:.4f}")

    return model, scaler, threshold_warn, threshold_crit


# ─── Vitals Autoencoder ──────────────────────────────────────────────

class VitalsAutoencoder(nn.Module):
    """
    Simple dense autoencoder for vitals feature vectors.
    Input/output: 4 features (hr_mean, hr_std, spo2_mean, spo2_std).
    Bottleneck forces learning of normal vitals distribution.
    """
    def __init__(self, input_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),   # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X: np.ndarray, name="VITALS"):
    """
    Train the vitals autoencoder on normal vitals readings.
    X: array of shape (n_samples, 4)
    Returns trained model, scaler, and reconstruction error threshold.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train / val split
    X_train, X_val = train_test_split(X_scaled, test_size=0.15, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train))
    val_ds   = TensorDataset(torch.tensor(X_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    model     = VitalsAutoencoder(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    EPOCHS     = 100
    best_val   = float("inf")
    best_state = None

    print(f"[{name}] Training autoencoder on {len(X_train)} samples...")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon    = model(batch)
                val_loss += criterion(recon, batch).item()

        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"train loss: {train_loss/len(train_loader):.6f} | "
                  f"val loss: {val_loss/len(val_loader):.6f}")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"[{name}] Best val loss: {best_val/len(val_loader):.6f}")

    # Compute reconstruction errors on full training set
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled)
        recon    = model(X_tensor)
        errors   = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

    threshold_warn = float(np.percentile(errors, 95))
    threshold_crit = float(np.percentile(errors, 99))

    print(f"[{name}] Reconstruction error range: "
          f"{errors.min():.6f} – {errors.max():.6f} (mean: {errors.mean():.6f})")
    print(f"[{name}] Warning threshold:  {threshold_warn:.6f}")
    print(f"[{name}] Critical threshold: {threshold_crit:.6f}")

    return model, scaler, threshold_warn, threshold_crit


# ─── Save and verify ─────────────────────────────────────────────────

def save_models(motion_model, motion_scaler, motion_tw, motion_tc,
                adl_model,    adl_scaler,    adl_tw,    adl_tc,
                vitals_model, vitals_scaler, vitals_tw, vitals_tc,
                df):

    joblib.dump(motion_model,  f"{MODELS_DIR}/motion_isoforest.joblib")
    joblib.dump(motion_scaler, f"{MODELS_DIR}/motion_scaler.joblib")
    joblib.dump(adl_model,     f"{MODELS_DIR}/adl_isoforest.joblib")
    joblib.dump(adl_scaler,    f"{MODELS_DIR}/adl_scaler.joblib")
    joblib.dump(vitals_scaler, f"{MODELS_DIR}/vitals_scaler.joblib")
    torch.save(vitals_model.state_dict(), f"{MODELS_DIR}/vitals_autoencoder.pt")

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "n_vectors":         len(df),
        "date_range":        [df["received_at"].iloc[0], df["received_at"].iloc[-1]],
        "motion_features":   MOTION_FEATURES,
        "adl_features":      ADL_FEATURES,
        "vitals_features":   VITALS_FEATURES,
        "thresholds": {
            "motion_warn":   motion_tw,
            "motion_crit":   motion_tc,
            "adl_warn":      adl_tw,
            "adl_crit":      adl_tc,
            "vitals_warn":   vitals_tw,
            "vitals_crit":   vitals_tc,
        },
        "fusion_weights": {
            "w_motion": 0.30,
            "w_adl":    0.35,
            "w_vitals": 0.35,
        }
    }

    with open(f"{MODELS_DIR}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] Models saved to ./{MODELS_DIR}/")
    print(f"[SAVE] Metadata saved to ./{MODELS_DIR}/training_meta.json")
    return meta


def verify_models(meta):
    """Quick sanity check — reload models and score one sample."""
    print("\n[VERIFY] Loading and testing saved models...")

    motion_model  = joblib.load(f"{MODELS_DIR}/motion_isoforest.joblib")
    motion_scaler = joblib.load(f"{MODELS_DIR}/motion_scaler.joblib")
    adl_model     = joblib.load(f"{MODELS_DIR}/adl_isoforest.joblib")
    adl_scaler    = joblib.load(f"{MODELS_DIR}/adl_scaler.joblib")
    vitals_scaler = joblib.load(f"{MODELS_DIR}/vitals_scaler.joblib")

    ae = VitalsAutoencoder(input_dim=len(VITALS_FEATURES))
    ae.load_state_dict(torch.load(f"{MODELS_DIR}/vitals_autoencoder.pt"))
    ae.eval()

    # Score a dummy normal sample
    normal_motion = np.array([[2, 0.3, 120, 0.5, 0.866]])   # pir_count=2, ratio=0.3, etc.
    normal_adl    = np.array([[0.1, 0.01, 1, 45, 2, 0.5, 0.866]])
    normal_vitals = np.array([[72.0, 3.0, 98.0, 0.5]])

    s_motion = score_isoforest(motion_model, motion_scaler, normal_motion)[0]
    s_adl    = score_isoforest(adl_model,    adl_scaler,    normal_adl)[0]
    s_vitals = score_autoencoder(ae, vitals_scaler, normal_vitals)[0]

    w = meta["fusion_weights"]
    s_global = w["w_motion"] * s_motion + w["w_adl"] * s_adl + w["w_vitals"] * s_vitals

    print(f"[VERIFY] Dummy normal sample scores:")
    print(f"         A_motion = {s_motion:.4f}  (warn > {meta['thresholds']['motion_warn']:.4f})")
    print(f"         A_adl    = {s_adl:.4f}  (warn > {meta['thresholds']['adl_warn']:.4f})")
    print(f"         A_vitals = {s_vitals:.4f}  (warn > {meta['thresholds']['vitals_warn']:.4f})")
    print(f"         A_global = {s_global:.4f}")
    print("[VERIFY] Models loaded and scoring correctly.")


# ─── Scoring helpers (reused in Phase 2 Step 3) ──────────────────────

def score_isoforest(model, scaler, X: np.ndarray) -> np.ndarray:
    """Normalised anomaly score in [0, 1]. Higher = more anomalous."""
    X_scaled   = scaler.transform(X)
    raw        = model.score_samples(X_scaled)  # range ≈ [-1, 0]
    return np.clip(1.0 + raw, 0.0, 1.0)         # shift to [0, 1]


def score_autoencoder(model, scaler, X: np.ndarray) -> np.ndarray:
    """Reconstruction error as anomaly score. Clipped to [0, 1] via threshold."""
    model.eval()
    X_scaled = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        tensor = torch.tensor(X_scaled)
        recon  = model(tensor)
        errors = ((tensor - recon) ** 2).mean(dim=1).numpy()
    return errors   # Raw MSE — normalise using vitals_crit threshold in scorer


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Dementia Monitor — Phase 2 Model Training")
    print("=" * 55)

    # Load
    df = load_data()

    # ── Motion model ──
    print("\n[1/3] Training motion anomaly model (Isolation Forest)...")
    X_motion = df[MOTION_FEATURES].values
    motion_model, motion_scaler, motion_tw, motion_tc = train_isolation_forest(
        X_motion, name="MOTION"
    )

    # ── ADL model ──
    print("\n[2/3] Training ADL anomaly model (Isolation Forest)...")
    X_adl = df[ADL_FEATURES].values
    adl_model, adl_scaler, adl_tw, adl_tc = train_isolation_forest(
        X_adl, name="ADL"
    )

    # ── Vitals autoencoder ──
    print("\n[3/3] Training vitals autoencoder...")
    vitals_df = df[df["vitals_valid"] == 1][VITALS_FEATURES]
    if len(vitals_df) < 20:
        print(f"[VITALS] Only {len(vitals_df)} valid vitals readings — "
              "need 20+ for autoencoder. Using Isolation Forest fallback.")
        vitals_model_type = "isoforest"
        vitals_model, vitals_scaler, vitals_tw, vitals_tc = train_isolation_forest(
            vitals_df.values, name="VITALS"
        )
        joblib.dump(vitals_model,  f"{MODELS_DIR}/vitals_isoforest.joblib")
        joblib.dump(vitals_scaler, f"{MODELS_DIR}/vitals_scaler.joblib")
    else:
        vitals_model_type = "autoencoder"
        vitals_model, vitals_scaler, vitals_tw, vitals_tc = train_autoencoder(
            vitals_df.values
        )

    # ── Save ──
    meta = save_models(
        motion_model, motion_scaler, motion_tw, motion_tc,
        adl_model,    adl_scaler,    adl_tw,    adl_tc,
        vitals_model, vitals_scaler, vitals_tw, vitals_tc,
        df,
    )
    meta["vitals_model_type"] = vitals_model_type

    # Re-save meta with vitals_model_type
    with open(f"{MODELS_DIR}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Verify ──
    if vitals_model_type == "autoencoder":
        verify_models(meta)

    print("\n[DONE] Training complete. Run online_scorer.py to start scoring.")


if __name__ == "__main__":
    main()