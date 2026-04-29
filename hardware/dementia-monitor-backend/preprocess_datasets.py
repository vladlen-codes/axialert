import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
WINDOW_MIN = 5        # 5-minute windows — matches ESP32
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Shared feature schema — matches ESP32 feature vector exactly ────
FEATURE_COLS = [
    "subject_id", "window_start",
    # Metadata
    "hour", "hour_sin", "hour_cos",
    # Motion
    "pir_count", "pir_ratio", "no_motion_s",
    # Appliance / ADL
    "cur_mean", "cur_std", "app_on", "app_on_s", "app_switches",
    # Vitals
    "hr_mean", "hr_std", "spo2_mean", "spo2_std",
    "tachy", "brady", "spo2_low", "vitals_valid",
    # Cross-modal
    "mot_and_app", "mot_no_app", "nomot_abnvit",
    # Source tag
    "source",
]

HR_HIGH  = 110.0
HR_LOW   =  50.0
SPO2_LOW =  92.0


def make_empty_row(subject_id, window_start, source):
    """Return a feature row with safe defaults for missing modalities."""
    h = window_start.hour
    return {
        "subject_id":   subject_id,
        "window_start": window_start.isoformat(),
        "hour":         h,
        "hour_sin":     float(np.sin(2 * np.pi * h / 24)),
        "hour_cos":     float(np.cos(2 * np.pi * h / 24)),
        # Motion — defaults: no motion
        "pir_count":    0,
        "pir_ratio":    0.0,
        "no_motion_s":  300.0,   # full window = 5 min
        # Appliance — defaults: off
        "cur_mean":     0.0,
        "cur_std":      0.0,
        "app_on":       0,
        "app_on_s":     0.0,
        "app_switches": 0,
        # Vitals — defaults: not measured
        "hr_mean":      0.0,
        "hr_std":       0.0,
        "spo2_mean":    0.0,
        "spo2_std":     0.0,
        "tachy":        0,
        "brady":        0,
        "spo2_low":     0,
        "vitals_valid": 0,
        # Cross-modal
        "mot_and_app":  0,
        "mot_no_app":   0,
        "nomot_abnvit": 0,
        "source":       source,
    }


def fill_cross_modal(row: dict) -> dict:
    """Compute cross-modal flags from already-filled motion + vitals."""
    pir_active      = row["pir_ratio"] > 0.05
    app_on          = row["app_on"] == 1
    abnormal_vitals = row["tachy"] or row["brady"] or row["spo2_low"]

    row["mot_and_app"]  = int(pir_active and app_on)
    row["mot_no_app"]   = int(pir_active and not app_on)
    row["nomot_abnvit"] = int(not pir_active and abnormal_vitals and row["vitals_valid"])
    return row


# ═══════════════════════════════════════════════════════════════════════
# CASAS PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_casas(path: str) -> pd.DataFrame:
    """
    Load a CASAS raw sensor event file.
    Handles both space-delimited and CSV formats.
    Returns a normalised DataFrame with columns:
        timestamp, sensor_id, sensor_type, value
    """
    print(f"[CASAS] Loading {path}")

    # Try CSV first, then whitespace-delimited
    try:
        df = pd.read_csv(path, header=None, sep=",")
        if df.shape[1] < 3:
            raise ValueError("Too few columns for CSV")
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")

    # CASAS format: date time sensor_id value [activity]
    if df.shape[1] >= 4:
        df.columns = (["date", "time", "sensor_id", "value"] +
                      [f"extra_{i}" for i in range(df.shape[1] - 4)])
        df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                                         errors="coerce")
    elif df.shape[1] == 3:
        df.columns = ["timestamp", "sensor_id", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"Unexpected CASAS format: {df.shape[1]} columns")

    df = df.dropna(subset=["timestamp"])
    df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
    df["value"]     = df["value"].astype(str).str.strip().str.upper()

    # Infer sensor type from ID prefix
    # M = motion, D = door, I = item/appliance, T = temperature, etc.
    df["sensor_type"] = df["sensor_id"].str[0]

    print(f"[CASAS] {len(df)} events | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"[CASAS] Sensor types: {df['sensor_type'].value_counts().to_dict()}")

    return df[["timestamp", "sensor_id", "sensor_type", "value"]]


def casas_to_windows(df: pd.DataFrame, subject_id="casas") -> pd.DataFrame:
    """
    Convert CASAS event stream to 5-minute feature windows.
    Motion sensors (M, D) → PIR features
    Appliance sensors (I, A) → ADL features
    """
    WINDOW = timedelta(minutes=WINDOW_MIN)

    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    t_start = df["timestamp"].min().floor("5min")
    t_end   = df["timestamp"].max().ceil("5min")

    motion_df    = df[df["sensor_type"].isin(["M", "D"])].copy()
    appliance_df = df[df["sensor_type"].isin(["I", "A"])].copy()

    rows = []
    current = t_start

    print(f"[CASAS] Windowing {subject_id}: "
          f"{int((t_end - t_start).total_seconds() / 300)} windows...")

    while current < t_end:
        win_end = current + WINDOW
        row = make_empty_row(subject_id, current, "casas")

        # ── Motion features ──
        w_motion = motion_df[
            (motion_df["timestamp"] >= current) &
            (motion_df["timestamp"] <  win_end)
        ]

        if not w_motion.empty:
            on_events  = w_motion[w_motion["value"] == "ON"]
            off_events = w_motion[w_motion["value"] == "OFF"]

            row["pir_count"] = len(on_events)

            # Estimate active duration by pairing ON/OFF events
            active_ms = 0
            for _, on_ev in on_events.iterrows():
                # Find nearest OFF after this ON within the window
                offs_after = off_events[off_events["timestamp"] > on_ev["timestamp"]]
                if not offs_after.empty:
                    off_t = offs_after.iloc[0]["timestamp"]
                    active_ms += (off_t - on_ev["timestamp"]).total_seconds()
                else:
                    active_ms += 30  # assume 30s active if no OFF seen

            window_s         = WINDOW.total_seconds()
            row["pir_ratio"] = min(active_ms / window_s, 1.0)

            last_trigger = w_motion["timestamp"].max()
            row["no_motion_s"] = max(0, (win_end - last_trigger).total_seconds())
        else:
            # Check time since last motion before this window
            prev_motion = motion_df[motion_df["timestamp"] < current]
            if not prev_motion.empty:
                last = prev_motion["timestamp"].max()
                row["no_motion_s"] = (win_end - last).total_seconds()
            else:
                row["no_motion_s"] = 300.0

        # ── Appliance features ──
        w_app = appliance_df[
            (appliance_df["timestamp"] >= current) &
            (appliance_df["timestamp"] <  win_end)
        ]

        if not w_app.empty:
            on_app  = w_app[w_app["value"] == "ON"]
            off_app = w_app[w_app["value"] == "OFF"]

            on_s = 0
            switches = 0
            prev_state = False

            for _, ev in w_app.sort_values("timestamp").iterrows():
                state = (ev["value"] == "ON")
                if state != prev_state:
                    switches += 1
                if state:
                    # Find matching OFF
                    offs = off_app[off_app["timestamp"] > ev["timestamp"]]
                    if not offs.empty:
                        on_s += (offs.iloc[0]["timestamp"] - ev["timestamp"]).total_seconds()
                    else:
                        on_s += 60  # assume 1 min if no OFF
                prev_state = state

            row["app_on"]       = int(len(on_app) > 0)
            row["app_on_s"]     = min(on_s, WINDOW.total_seconds())
            row["app_switches"] = switches
            # Simulate cur_mean from ON duration (no actual current in CASAS)
            row["cur_mean"] = 0.5 * (row["app_on_s"] / WINDOW.total_seconds())
            row["cur_std"]  = 0.1

        row = fill_cross_modal(row)
        rows.append(row)
        current = win_end

    result = pd.DataFrame(rows, columns=FEATURE_COLS)
    print(f"[CASAS] {len(result)} windows generated for {subject_id}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# TIHM PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_tihm_activity(path: str) -> pd.DataFrame:
    """
    Load TIHM activity/motion sensor data.
    Expected columns: patient_id, timestamp, location, event_type
    (exact column names vary by TIHM release — we auto-detect)
    """
    print(f"[TIHM] Loading activity data from {path}")
    df = pd.read_csv(path)
    print(f"[TIHM] Activity columns: {list(df.columns)}")

    # Auto-detect column names (TIHM has had several formats)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if any(x in cl for x in ["patient", "subject", "id"]):
            col_map["patient_id"] = col
        elif any(x in cl for x in ["time", "date", "stamp"]):
            col_map["timestamp"] = col
        elif any(x in cl for x in ["location", "room", "sensor", "device"]):
            col_map["location"] = col
        elif any(x in cl for x in ["event", "type", "value", "state"]):
            col_map["event_type"] = col

    missing = [k for k in ["patient_id", "timestamp"] if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not auto-detect columns {missing} in TIHM activity file.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please rename columns to: patient_id, timestamp, location, event_type"
        )

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    print(f"[TIHM] {len(df)} activity events | "
          f"{df['patient_id'].nunique()} patients")
    return df


def load_tihm_physiology(path: str) -> pd.DataFrame:
    """
    Load TIHM physiology data (HR + SpO2).
    Expected columns: patient_id, timestamp, heart_rate, spo2
    """
    print(f"[TIHM] Loading physiology data from {path}")
    df = pd.read_csv(path)
    print(f"[TIHM] Physiology columns: {list(df.columns)}")

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if any(x in cl for x in ["patient", "subject", "id"]):
            col_map["patient_id"] = col
        elif any(x in cl for x in ["time", "date", "stamp"]):
            col_map["timestamp"] = col
        elif any(x in cl for x in ["heart", "hr", "pulse", "bpm"]):
            col_map["heart_rate"] = col
        elif any(x in cl for x in ["spo2", "oxygen", "saturation", "o2"]):
            col_map["spo2"] = col

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "heart_rate" in df.columns:
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    if "spo2" in df.columns:
        df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")

    print(f"[TIHM] {len(df)} physiology readings | "
          f"{df['patient_id'].nunique()} patients")
    return df


def tihm_to_windows(activity_df: pd.DataFrame,
                    physiology_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert TIHM event streams to 5-minute feature windows.
    Processes each patient separately, then concatenates.
    """
    WINDOW = timedelta(minutes=WINDOW_MIN)
    all_rows = []

    patients = activity_df["patient_id"].unique()
    print(f"[TIHM] Processing {len(patients)} patients...")

    for pid in tqdm(patients, desc="TIHM patients"):
        pat_activity = activity_df[activity_df["patient_id"] == pid].copy()

        pat_physio = None
        if physiology_df is not None and "patient_id" in physiology_df.columns:
            pat_physio = physiology_df[physiology_df["patient_id"] == pid].copy()

        t_start = pat_activity["timestamp"].min().floor("5min")
        t_end   = pat_activity["timestamp"].max().ceil("5min")
        current = t_start

        while current < t_end:
            win_end = current + WINDOW
            row = make_empty_row(str(pid), current, "tihm")

            # ── Motion features ──
            w_act = pat_activity[
                (pat_activity["timestamp"] >= current) &
                (pat_activity["timestamp"] <  win_end)
            ]

            if not w_act.empty:
                row["pir_count"] = len(w_act)
                # Estimate ratio from event density (each event ≈ 30s of presence)
                estimated_active_s = min(len(w_act) * 30, WINDOW.total_seconds())
                row["pir_ratio"]   = estimated_active_s / WINDOW.total_seconds()
                last_ev = w_act["timestamp"].max()
                row["no_motion_s"] = (win_end - last_ev).total_seconds()
            else:
                prev = pat_activity[pat_activity["timestamp"] < current]
                if not prev.empty:
                    row["no_motion_s"] = (win_end - prev["timestamp"].max()).total_seconds()

            # ── Vitals features ──
            if pat_physio is not None and not pat_physio.empty:
                w_phys = pat_physio[
                    (pat_physio["timestamp"] >= current) &
                    (pat_physio["timestamp"] <  win_end)
                ]

                hr_vals   = w_phys["heart_rate"].dropna().values if "heart_rate" in w_phys.columns else []
                spo2_vals = w_phys["spo2"].dropna().values       if "spo2"       in w_phys.columns else []

                if len(hr_vals) > 0 and len(spo2_vals) > 0:
                    row["hr_mean"]      = float(np.mean(hr_vals))
                    row["hr_std"]       = float(np.std(hr_vals))   if len(hr_vals)   > 1 else 0.0
                    row["spo2_mean"]    = float(np.mean(spo2_vals))
                    row["spo2_std"]     = float(np.std(spo2_vals)) if len(spo2_vals) > 1 else 0.0
                    row["tachy"]        = int(row["hr_mean"]   > HR_HIGH)
                    row["brady"]        = int(row["hr_mean"]   < HR_LOW and row["hr_mean"] > 0)
                    row["spo2_low"]     = int(row["spo2_mean"] < SPO2_LOW and row["spo2_mean"] > 0)
                    row["vitals_valid"] = 1

            row = fill_cross_modal(row)
            all_rows.append(row)
            current = win_end

    result = pd.DataFrame(all_rows, columns=FEATURE_COLS)
    print(f"[TIHM] {len(result)} total windows across {len(patients)} patients")
    return result


# ═══════════════════════════════════════════════════════════════════════
# OPPORTUNITY DATASET PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

# Columns in the .dat files (0-indexed)
_OPP_MS_COL    = 0
_OPP_LOCO_COL  = 243   # Locomotion: 0=None,1=Stand,2=Walk,4=Sit,5=Lie
_OPP_REED_COLS = list(range(194, 207))  # Reed switch binary sensors
_OPP_HZ        = 30


def load_opportunity(directory: str) -> list[tuple[str, pd.DataFrame]]:
    """
    Load all OPPORTUNITY .dat files from a directory.
    Returns list of (subject_session_id, df) tuples where df has
    columns: timestamp, locomotion, reed_any.
    """
    import glob
    files = sorted(glob.glob(os.path.join(directory, "S*.dat")))
    if not files:
        raise FileNotFoundError(f"No S*.dat files found in {directory}")

    print(f"[OPP] Found {len(files)} session files in {directory}")
    sessions = []
    base_date = datetime(2024, 1, 1)  # arbitrary anchor date

    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]  # e.g. S1-ADL1
        df = pd.read_csv(fpath, sep=" ", header=None, na_values="NaN",
                         usecols=[_OPP_MS_COL, _OPP_LOCO_COL] + _OPP_REED_COLS,
                         low_memory=False)

        df.columns = ["ms", "locomotion"] + [f"reed_{i}" for i in range(len(_OPP_REED_COLS))]
        df["timestamp"] = pd.to_datetime(
            df["ms"].apply(lambda ms: base_date + timedelta(milliseconds=float(ms))),
            errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])
        df["locomotion"] = pd.to_numeric(df["locomotion"], errors="coerce").fillna(0).astype(int)
        for rc in [c for c in df.columns if c.startswith("reed_")]:
            df[rc] = pd.to_numeric(df[rc], errors="coerce").fillna(0).astype(int)
        df["reed_any"] = df[[c for c in df.columns if c.startswith("reed_")]].max(axis=1)

        # Advance base_date so sessions don't overlap
        duration_s = df["ms"].max() / 1000
        base_date += timedelta(seconds=duration_s + 3600)

        sessions.append((name, df[["timestamp", "locomotion", "reed_any"]]))
        print(f"[OPP]   {name}: {len(df)} samples, loco={df['locomotion'].value_counts().to_dict()}")

    return sessions


def opportunity_to_windows(sessions: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Convert OPPORTUNITY sessions to 5-minute feature windows.
    Locomotion Walk(2)/Stand(1) → PIR active.
    Reed switches → appliance on/off.
    """
    WINDOW = timedelta(minutes=WINDOW_MIN)
    all_rows = []

    for session_id, df in sessions:
        t_start = df["timestamp"].min().floor("5min")
        t_end   = df["timestamp"].max().ceil("5min")
        current = t_start

        while current < t_end:
            win_end = current + WINDOW
            row = make_empty_row(session_id, current, "opportunity")

            w = df[(df["timestamp"] >= current) & (df["timestamp"] < win_end)]
            if not w.empty:
                # Each sample = 1/30 s; active = walk(2) or stand(1)
                sample_s    = 1.0 / _OPP_HZ
                active_mask = w["locomotion"].isin([1, 2])
                active_s    = active_mask.sum() * sample_s
                window_s    = WINDOW.total_seconds()

                row["pir_count"] = int(active_mask.sum())
                row["pir_ratio"] = min(active_s / window_s, 1.0)

                last_active = w[active_mask]["timestamp"].max() if active_mask.any() else None
                row["no_motion_s"] = (win_end - last_active).total_seconds() if last_active else window_s

                # Appliance: reed switch on for at least one sample
                reed_on_s = w["reed_any"].sum() * sample_s
                row["app_on"]       = int(w["reed_any"].any())
                row["app_on_s"]     = min(reed_on_s, window_s)
                row["app_switches"] = int((w["reed_any"].diff().abs().fillna(0) > 0).sum())
                row["cur_mean"]     = 0.5 * (reed_on_s / window_s)
                row["cur_std"]      = 0.1 if row["app_on"] else 0.0

            row = fill_cross_modal(row)
            all_rows.append(row)
            current = win_end

    result = pd.DataFrame(all_rows, columns=FEATURE_COLS)
    print(f"[OPP] {len(result)} windows from {len(sessions)} sessions")
    return result


# ═══════════════════════════════════════════════════════════════════════
# MIMIC-III DEMO VITALS PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_mimic_vitals(path: str) -> pd.DataFrame:
    """
    Load MIMIC-III demo vitals CSV (patient_id, timestamp, heart_rate, spo2).
    Returns pretrain-ready DataFrame with hr_mean, hr_std, spo2_mean, spo2_std.
    """
    print(f"[MIMIC] Loading {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["spo2"]       = pd.to_numeric(df["spo2"],       errors="coerce")

    # Aggregate into 5-minute windows per patient
    df = df.set_index("timestamp").sort_index()
    rows = []
    for _, pat in df.groupby("patient_id"):
        resampled = pat.resample("5min").agg(
            hr_mean=("heart_rate", "mean"),
            hr_std=("heart_rate", "std"),
            spo2_mean=("spo2", "mean"),
            spo2_std=("spo2", "std"),
        ).dropna(subset=["hr_mean", "spo2_mean"])
        rows.append(resampled)

    if not rows:
        return pd.DataFrame(columns=["hr_mean", "hr_std", "spo2_mean", "spo2_std"])

    result = pd.concat(rows).reset_index(drop=True)
    result["hr_std"]   = result["hr_std"].fillna(0.0)
    result["spo2_std"] = result["spo2_std"].fillna(0.0)
    print(f"[MIMIC] {len(result)} 5-min vitals windows | "
          f"HR {result['hr_mean'].mean():.1f}±{result['hr_mean'].std():.1f} | "
          f"SpO2 {result['spo2_mean'].mean():.1f}±{result['spo2_mean'].std():.1f}")
    return result[["hr_mean", "hr_std", "spo2_mean", "spo2_std"]]


# ═══════════════════════════════════════════════════════════════════════
# KAGGLE VITALS PREPROCESSING (dementia_patients_health_data.csv)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_kaggle_vitals(path: str) -> pd.DataFrame:
    """
    Extract and map vitals from the Kaggle dementia dataset
    for autoencoder pre-training.
    Maps HeartRate → hr_mean, BloodOxygenLevel → spo2_mean.
    hr_std and spo2_std are estimated from population std as placeholders.
    """
    print(f"[KAGGLE] Loading {path}")
    df = pd.read_csv(path)

    hr_std_est   = df["HeartRate"].std() * 0.1         # 10% of pop std as within-window estimate
    spo2_std_est = df["BloodOxygenLevel"].std() * 0.1

    vitals = pd.DataFrame({
        "hr_mean":   df["HeartRate"],
        "hr_std":    hr_std_est,
        "spo2_mean": df["BloodOxygenLevel"],
        "spo2_std":  spo2_std_est,
    })
    vitals = vitals.dropna()
    print(f"[KAGGLE] {len(vitals)} vitals rows extracted")
    return vitals


# ═══════════════════════════════════════════════════════════════════════
# COMBINE AND SAVE
# ═══════════════════════════════════════════════════════════════════════

def combine_and_save(casas_df: pd.DataFrame = None,
                     tihm_df:  pd.DataFrame = None,
                     opp_df:   pd.DataFrame = None,
                     kaggle_vitals:  pd.DataFrame = None,
                     mimic_vitals:   pd.DataFrame = None):
    """
    Merge processed datasets, compute stats, and save to output files.
    """
    parts = []
    if casas_df is not None and not casas_df.empty:
        parts.append(casas_df)
        casas_df.to_csv(f"{OUTPUT_DIR}/casas_features.csv", index=False)
        print(f"[SAVE] casas_features.csv — {len(casas_df)} rows")

    if tihm_df is not None and not tihm_df.empty:
        parts.append(tihm_df)
        tihm_df.to_csv(f"{OUTPUT_DIR}/tihm_features.csv", index=False)
        print(f"[SAVE] tihm_features.csv — {len(tihm_df)} rows")

    if opp_df is not None and not opp_df.empty:
        parts.append(opp_df)
        opp_df.to_csv(f"{OUTPUT_DIR}/opportunity_features.csv", index=False)
        print(f"[SAVE] opportunity_features.csv — {len(opp_df)} rows")

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(f"{OUTPUT_DIR}/combined_features.csv", index=False)
        print(f"\n[SAVE] combined_features.csv — {len(combined)} total rows")
        print(f"       Subjects: {combined['subject_id'].nunique()}")
        print(f"       Sources:  {combined['source'].value_counts().to_dict()}")
        print(f"       Windows with vitals: {combined['vitals_valid'].sum()}")
        print(f"       Windows with motion: {(combined['pir_count'] > 0).sum()}")
        print(f"       Windows with appliance ON: {combined['app_on'].sum()}")
        print()
        print("[SAVE] Numeric summary:")
        print(combined[[
            "pir_count", "pir_ratio", "no_motion_s",
            "cur_mean", "app_on_s",
            "hr_mean", "spo2_mean"
        ]].describe().round(3).to_string())

    # Save vitals pre-training set
    vitals_parts = []
    if tihm_df is not None:
        tv = tihm_df[tihm_df["vitals_valid"] == 1][
            ["hr_mean", "hr_std", "spo2_mean", "spo2_std"]
        ]
        vitals_parts.append(tv)

    if kaggle_vitals is not None:
        vitals_parts.append(kaggle_vitals)

    if mimic_vitals is not None:
        vitals_parts.append(mimic_vitals)

    if vitals_parts:
        pretrain = pd.concat(vitals_parts, ignore_index=True).dropna()
        pretrain.to_csv(f"{OUTPUT_DIR}/pretrain_vitals.csv", index=False)
        print(f"\n[SAVE] pretrain_vitals.csv — {len(pretrain)} rows "
              f"(for autoencoder pre-training)")

    print(f"\n[DONE] All files saved to ./{OUTPUT_DIR}/")
    print("       Next step: update DB_PATH in train_models.py to use combined_features.csv")
    print("       OR copy rows into your SQLite DB with:")
    print("       python preprocess_datasets.py --import-to-db")


def import_to_db(combined_csv: str, db_path: str = "dementia_monitor.db"):
    """
    Import combined_features.csv into the SQLite database
    so train_models.py can use it directly without modification.
    """
    import sqlite3
    print(f"[DB] Importing {combined_csv} into {db_path}")

    df = pd.read_csv(combined_csv)
    conn = sqlite3.connect(db_path)

    # Map combined CSV columns to DB schema
    df_db = df.copy()  # FIX #25: was a no-op rename — removed
    df_db["received_at"] = df_db["window_start"]
    df_db["ts"] = 0   # no ESP32 millis for external data

    cols = [
        "received_at", "ts", "hour", "hour_sin", "hour_cos",
        "pir_count", "pir_ratio", "no_motion_s",
        "cur_mean", "cur_std", "app_on", "app_on_s", "app_switches",
        "hr_mean", "hr_std", "spo2_mean", "spo2_std",
        "tachy", "brady", "spo2_low", "vitals_valid",
        "mot_and_app", "mot_no_app", "nomot_abnvit",
    ]
    df_db = df_db[[c for c in cols if c in df_db.columns]]

    df_db.to_sql("feature_vectors", conn, if_exists="append", index=False)
    count = conn.execute("SELECT COUNT(*) FROM feature_vectors").fetchone()[0]
    conn.close()

    print(f"[DB] Imported {len(df_db)} rows. Total in DB: {count}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Preprocess CASAS / TIHM / OPPORTUNITY / MIMIC datasets")
    parser.add_argument("--casas",            default=os.path.join(BASE_DIR, "data", "casas", "casas_raw.csv"))
    parser.add_argument("--tihm-activity",    default=os.path.join(BASE_DIR, "data", "tihm", "tihm_activity.csv"))
    parser.add_argument("--tihm-physiology",  default=os.path.join(BASE_DIR, "data", "tihm", "tihm_physiology.csv"))
    parser.add_argument("--opportunity",      default=os.path.join(BASE_DIR, "data", "opportunity"),
                        help="Directory containing OPPORTUNITY .dat files")
    parser.add_argument("--mimic-vitals",     default=os.path.join(BASE_DIR, "data", "mimic", "mimic_vitals.csv"),
                        help="MIMIC-III demo vitals CSV (patient_id, timestamp, heart_rate, spo2)")
    parser.add_argument("--include-kaggle",   default=None,
                        help="Path to dementia_patients_health_data.csv")
    parser.add_argument("--casas-only",       action="store_true")
    parser.add_argument("--tihm-only",        action="store_true")
    parser.add_argument("--import-to-db",     action="store_true",
                        help="Import combined_features.csv into SQLite DB")
    parser.add_argument("--db",               default=os.path.join(BASE_DIR, "dementia_monitor.db"))
    args = parser.parse_args()

    # Import mode
    if args.import_to_db:
        import_to_db(f"{OUTPUT_DIR}/combined_features.csv", args.db)
        return

    print("=" * 60)
    print("  Dementia Monitor — Dataset Preprocessing Pipeline")
    print("=" * 60)

    casas_df      = None
    tihm_df       = None
    opp_df        = None
    kaggle_vitals = None
    mimic_vitals  = None

    # ── CASAS ──
    if not args.tihm_only:
        if os.path.exists(args.casas):
            try:
                casas_raw = load_casas(args.casas)
                casas_df  = casas_to_windows(casas_raw)
            except Exception as e:
                print(f"[CASAS] Error: {e}")
                print("[CASAS] Skipping. Check file format matches expected CASAS layout.")
        else:
            print(f"[CASAS] File not found: {args.casas}")
            print("[CASAS] Download from https://casas.wsu.edu/datasets/ and place at that path.")

    # ── TIHM ──
    if not args.casas_only:
        if os.path.exists(args.tihm_activity):
            try:
                tihm_act  = load_tihm_activity(args.tihm_activity)
                tihm_phys = None
                if os.path.exists(args.tihm_physiology):
                    tihm_phys = load_tihm_physiology(args.tihm_physiology)
                else:
                    print(f"[TIHM] Physiology file not found: {args.tihm_physiology} — "
                          "processing motion only")
                tihm_df = tihm_to_windows(tihm_act, tihm_phys)
            except Exception as e:
                print(f"[TIHM] Error: {e}")
                print("[TIHM] Skipping. Check file format.")
        else:
            print(f"[TIHM] File not found: {args.tihm_activity}")
            print("[TIHM] Request access at https://www.synapse.org/#!Synapse:syn26479697")

    # ── OPPORTUNITY ──
    if os.path.isdir(args.opportunity):
        try:
            sessions = load_opportunity(args.opportunity)
            opp_df   = opportunity_to_windows(sessions)
        except Exception as e:
            print(f"[OPP] Error: {e}")

    # ── MIMIC vitals ──
    if os.path.exists(args.mimic_vitals):
        try:
            mimic_vitals = load_mimic_vitals(args.mimic_vitals)
        except Exception as e:
            print(f"[MIMIC] Error: {e}")

    # ── Kaggle vitals ──
    if args.include_kaggle and os.path.exists(args.include_kaggle):
        try:
            kaggle_vitals = preprocess_kaggle_vitals(args.include_kaggle)
        except Exception as e:
            print(f"[KAGGLE] Error: {e}")

    # ── Nothing loaded ──
    if casas_df is None and tihm_df is None and opp_df is None:
        print("\n[ERROR] No datasets found. Nothing to process.")
        print("        Place dataset files at the expected paths and re-run.")
        print("        See the file header for download instructions.")
        sys.exit(1)

    combine_and_save(casas_df, tihm_df, opp_df, kaggle_vitals, mimic_vitals)


if __name__ == "__main__":
    main()