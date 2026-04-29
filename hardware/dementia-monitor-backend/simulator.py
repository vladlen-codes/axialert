#!/usr/bin/env python3
"""
simulator.py — Patient behaviour simulator for Axialert
========================================================
Replaces a real ESP32 + patient by POSTing synthetic feature vectors
directly to the ingestion server.  Use this for end-to-end testing
without physical hardware or a real patient.

USAGE
-----
  # Single scenario (posts ~12 vectors then exits)
  python simulator.py --scenario normal_day
  python simulator.py --scenario prolonged_inactivity
  python simulator.py --scenario night_wandering
  python simulator.py --scenario tachycardia_event
  python simulator.py --scenario low_spo2_event
  python simulator.py --scenario normal_night

  # All scenarios back-to-back
  python simulator.py --scenario all

  # Seed 2 days of realistic synthetic data into the DB instantly
  python simulator.py --bulk-days 2

  # Skip inter-vector delays (CI / fast seeding)
  python simulator.py --scenario all --fast
  python simulator.py --bulk-days 2 --fast

  # Custom server URL
  python simulator.py --scenario all --url http://192.168.1.5:7777

  # List available scenarios
  python simulator.py --list

REQUIREMENTS
------------
  pip install requests
  Stack must be running: ./run_stack.sh
"""

import argparse
import math
import random
import sys
import time
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    sys.exit("[simulator] ERROR: 'requests' not installed. Run: pip install requests")

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_URL    = "http://127.0.0.1:7777"
VECTORS_URL    = "{base}/vectors"
INGEST_URL     = "{base}/ingest"
INTER_DELAY_S  = 1.0   # pause between vectors in normal mode
POLL_WAIT_S    = 35    # seconds to wait for scorer to label a vector

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _trig(hour: int):
    """Return (hour_sin, hour_cos) for a given hour."""
    return (
        round(math.sin(2 * math.pi * hour / 24), 4),
        round(math.cos(2 * math.pi * hour / 24), 4),
    )


def make_vector(hour: int, **overrides) -> dict:
    """
    Build a complete feature vector dict with sensible defaults.
    Cross-modal flags (mot_and_app, mot_no_app, nomot_abnvit) are
    computed automatically from the other fields.
    """
    hs, hc = _trig(hour)
    v = {
        "ts":           int(time.time() * 1000),
        "hour":         hour,
        "hour_sin":     hs,
        "hour_cos":     hc,
        # Motion
        "pir_count":    0,
        "pir_ratio":    0.0,
        "no_motion_s":  300.0,
        # Appliance / ADL
        "cur_mean":     0.0,
        "cur_std":      0.0,
        "app_on":       0,
        "app_on_s":     0.0,
        "app_switches": 0,
        # Vitals
        "hr_mean":      0.0,
        "hr_std":       0.0,
        "spo2_mean":    0.0,
        "spo2_std":     0.0,
        "tachy":        0,
        "brady":        0,
        "spo2_low":     0,
        "vitals_valid": 0,
        # Cross-modal — auto-computed below
        "mot_and_app":  0,
        "mot_no_app":   0,
        "nomot_abnvit": 0,
    }
    v.update(overrides)

    # Auto-compute cross-modal flags to keep them consistent
    pir_active      = v["pir_ratio"] > 0.05
    app_on          = v["app_on"] == 1
    abnormal_vitals = bool(v["tachy"] or v["brady"] or v["spo2_low"])
    v["mot_and_app"]  = 1 if (pir_active and app_on) else 0
    v["mot_no_app"]   = 1 if (pir_active and not app_on) else 0
    v["nomot_abnvit"] = 1 if (not pir_active and abnormal_vitals and v["vitals_valid"]) else 0

    return v


def _post(base_url: str, vector: dict) -> requests.Response:
    url = INGEST_URL.format(base=base_url)
    return requests.post(url, json=vector, timeout=5)


def _poll_label(base_url: str, vector_id: int, timeout_s: float = POLL_WAIT_S) -> str:
    """Poll /vectors until vector_id is scored, return its anomaly_label."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2)
        resp = requests.get(VECTORS_URL.format(base=base_url) + "?n=5", timeout=5)
        if resp.ok:
            for row in resp.json():
                if row["id"] == vector_id and row.get("anomaly_label"):
                    return row["anomaly_label"]
    return "PENDING (scorer not yet run)"


def _send_scenario(name: str, vectors: list, base_url: str, fast: bool):
    """Post a list of vectors and print scored results."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {name}")
    print(f"{'='*60}")

    for i, v in enumerate(vectors, 1):
        try:
            resp = _post(base_url, v)
            if resp.status_code in (200, 201):
                vid = resp.json().get("vector_id", "?")
                alerts = resp.json().get("alerts", [])
                rule_alerts = f"  rule-alerts={alerts}" if alerts else ""
                print(f"  [{i:02d}/{len(vectors)}] hour={v['hour']:02d}h "
                      f"PIR={v['pir_ratio']:.2f} HR={v['hr_mean']:.0f} "
                      f"SpO2={v['spo2_mean']:.0f} → id={vid}{rule_alerts}")
                if not fast and i == len(vectors):
                    label = _poll_label(base_url, vid)
                    sev   = label.split(":")[0]
                    icon  = "🔴" if sev == "CRITICAL" else "🟡" if sev == "WARNING" else "🟢"
                    print(f"  {icon} Scored label: {label}")
            else:
                print(f"  [{i:02d}] ERROR {resp.status_code}: {resp.text[:80]}")
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot reach {base_url}. Is the stack running?")
            print("  Run: cd hardware/dementia-monitor-backend && ./run_stack.sh")
            return

        if not fast:
            time.sleep(INTER_DELAY_S)


# ─── Scenario definitions ─────────────────────────────────────────────────────

def scenario_normal_day(base_url: str, fast: bool):
    """Healthy daytime routine — NORMAL expected."""
    hour = 10
    vectors = [
        make_vector(hour,
                    pir_count=10, pir_ratio=round(random.uniform(0.3, 0.6), 2),
                    no_motion_s=round(random.uniform(20, 90), 1),
                    cur_mean=round(random.uniform(0.3, 0.7), 3),
                    cur_std=round(random.uniform(0.05, 0.15), 3),
                    app_on=1,
                    app_on_s=round(random.uniform(80, 200), 1),
                    app_switches=random.randint(1, 4),
                    hr_mean=round(random.uniform(65, 80), 1),
                    hr_std=round(random.uniform(2, 5), 1),
                    spo2_mean=round(random.uniform(97, 99), 1),
                    spo2_std=round(random.uniform(0.2, 0.8), 1),
                    vitals_valid=1)
        for _ in range(12)
    ]
    _send_scenario("normal_day  [expect: NORMAL]", vectors, base_url, fast)


def scenario_normal_night(base_url: str, fast: bool):
    """Patient sleeping — NORMAL expected."""
    vectors = [
        make_vector(random.randint(1, 5),
                    pir_count=0, pir_ratio=0.0, no_motion_s=300.0,
                    cur_mean=0.0, cur_std=0.0, app_on=0,
                    hr_mean=round(random.uniform(52, 62), 1),
                    hr_std=round(random.uniform(1, 3), 1),
                    spo2_mean=round(random.uniform(95, 98), 1),
                    spo2_std=round(random.uniform(0.2, 0.5), 1),
                    vitals_valid=1)
        for _ in range(12)
    ]
    _send_scenario("normal_night  [expect: NORMAL]", vectors, base_url, fast)


def scenario_prolonged_inactivity(base_url: str, fast: bool):
    """No motion for 2+ hrs during daytime — CRITICAL:PROLONGED_INACTIVITY expected."""
    hour = 14
    vectors = [
        make_vector(hour,
                    pir_count=0, pir_ratio=0.0,
                    no_motion_s=round(7200 + i * 300, 1),   # growing inactivity
                    cur_mean=0.35, cur_std=0.05,
                    app_on=1, app_on_s=round(2400 + i * 300, 1),  # appliance left on
                    app_switches=1,
                    hr_mean=0.0, spo2_mean=0.0, vitals_valid=0)
        for i in range(12)
    ]
    _send_scenario("prolonged_inactivity  [expect: CRITICAL:PROLONGED_INACTIVITY]",
                   vectors, base_url, fast)


def scenario_night_wandering(base_url: str, fast: bool):
    """Motion at 3 AM, no appliance — WARNING:NIGHT_WANDERING expected."""
    hour = 3
    vectors = [
        make_vector(hour,
                    pir_count=random.randint(8, 18),
                    pir_ratio=round(random.uniform(0.3, 0.6), 2),
                    no_motion_s=round(random.uniform(10, 60), 1),
                    cur_mean=0.0, cur_std=0.0, app_on=0,
                    hr_mean=round(random.uniform(68, 85), 1),
                    hr_std=round(random.uniform(3, 6), 1),
                    spo2_mean=round(random.uniform(95, 98), 1),
                    spo2_std=0.4, vitals_valid=1)
        for _ in range(12)
    ]
    _send_scenario("night_wandering  [expect: WARNING:NIGHT_WANDERING]",
                   vectors, base_url, fast)


def scenario_tachycardia_event(base_url: str, fast: bool):
    """Elevated heart rate — WARNING:PHYSIOLOGICAL expected."""
    hour = 11
    vectors = [
        make_vector(hour,
                    pir_count=2, pir_ratio=0.08, no_motion_s=200.0,
                    cur_mean=0.1, cur_std=0.02, app_on=0,
                    hr_mean=round(random.uniform(118, 135), 1),
                    hr_std=round(random.uniform(4, 8), 1),
                    spo2_mean=round(random.uniform(94, 97), 1),
                    spo2_std=0.5,
                    tachy=1,
                    vitals_valid=1)
        for _ in range(12)
    ]
    _send_scenario("tachycardia_event  [expect: WARNING:PHYSIOLOGICAL or CRITICAL]",
                   vectors, base_url, fast)


def scenario_low_spo2_event(base_url: str, fast: bool):
    """Critical SpO2 drop + no motion — CRITICAL:PHYSIOLOGICAL_EVENT expected."""
    hour = 10
    vectors = [
        make_vector(hour,
                    pir_count=0, pir_ratio=0.0, no_motion_s=1200.0,
                    cur_mean=0.0, cur_std=0.0, app_on=0,
                    hr_mean=round(random.uniform(112, 125), 1),
                    hr_std=round(random.uniform(5, 10), 1),
                    spo2_mean=round(random.uniform(82, 88), 1),
                    spo2_std=round(random.uniform(1.0, 2.5), 1),
                    tachy=1, spo2_low=1,
                    vitals_valid=1)
        for _ in range(12)
    ]
    _send_scenario("low_spo2_event  [expect: CRITICAL:PHYSIOLOGICAL_EVENT]",
                   vectors, base_url, fast)


SCENARIOS = {
    "normal_day":            scenario_normal_day,
    "normal_night":          scenario_normal_night,
    "prolonged_inactivity":  scenario_prolonged_inactivity,
    "night_wandering":       scenario_night_wandering,
    "tachycardia_event":     scenario_tachycardia_event,
    "low_spo2_event":        scenario_low_spo2_event,
}

# ─── Bulk day generator ───────────────────────────────────────────────────────

def _daily_pattern(day_offset: int, anomaly_chance: float = 0.07) -> list:
    """
    Generate one full day of synthetic feature vectors (288 windows × 5 min).
    Follows a realistic daily activity cycle with occasional random anomalies.
    """
    vectors = []
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_offset)

    for window_idx in range(288):          # 24 h × 12 windows/h
        dt   = base_date + timedelta(minutes=window_idx * 5)
        hour = dt.hour
        ts   = int(dt.timestamp() * 1000)

        # Activity profile by hour
        if 0 <= hour < 6:                  # deep sleep
            pir_count = 0
            pir_ratio = 0.0
            no_motion = 300.0
            app_on    = 0
            app_on_s  = 0.0
            cur_mean  = 0.0
            hr        = round(random.gauss(55, 3), 1)
            spo2      = round(random.gauss(96.5, 0.5), 1)
            valid     = 1

        elif 6 <= hour < 8:                # waking up / morning prep
            pir_count = random.randint(3, 10)
            pir_ratio = round(random.uniform(0.15, 0.45), 3)
            no_motion = round(random.uniform(30, 120), 1)
            app_on    = 1
            app_on_s  = round(random.uniform(60, 180), 1)
            cur_mean  = round(random.uniform(0.2, 0.6), 3)
            hr        = round(random.gauss(70, 5), 1)
            spo2      = round(random.gauss(98, 0.5), 1)
            valid     = 1

        elif 8 <= hour < 12:               # active morning
            pir_count = random.randint(6, 15)
            pir_ratio = round(random.uniform(0.3, 0.65), 3)
            no_motion = round(random.uniform(10, 80), 1)
            app_on    = random.choice([0, 1])
            app_on_s  = round(random.uniform(0, 240), 1) if app_on else 0.0
            cur_mean  = round(random.uniform(0.1, 0.7), 3)
            hr        = round(random.gauss(72, 6), 1)
            spo2      = round(random.gauss(98, 0.5), 1)
            valid     = random.choice([0, 1])

        elif 12 <= hour < 14:              # lunch
            pir_count = random.randint(4, 12)
            pir_ratio = round(random.uniform(0.2, 0.5), 3)
            no_motion = round(random.uniform(20, 100), 1)
            app_on    = 1
            app_on_s  = round(random.uniform(60, 200), 1)
            cur_mean  = round(random.uniform(0.3, 0.7), 3)
            hr        = round(random.gauss(73, 5), 1)
            spo2      = round(random.gauss(97.5, 0.5), 1)
            valid     = 1

        elif 14 <= hour < 18:              # afternoon
            pir_count = random.randint(2, 10)
            pir_ratio = round(random.uniform(0.1, 0.4), 3)
            no_motion = round(random.uniform(30, 180), 1)
            app_on    = random.choice([0, 0, 1])
            app_on_s  = round(random.uniform(0, 150), 1) if app_on else 0.0
            cur_mean  = round(random.uniform(0.0, 0.5), 3)
            hr        = round(random.gauss(70, 5), 1)
            spo2      = round(random.gauss(97.5, 0.6), 1)
            valid     = random.choice([0, 0, 1])

        elif 18 <= hour < 22:              # evening / dinner
            pir_count = random.randint(3, 10)
            pir_ratio = round(random.uniform(0.15, 0.45), 3)
            no_motion = round(random.uniform(20, 120), 1)
            app_on    = 1
            app_on_s  = round(random.uniform(60, 250), 1)
            cur_mean  = round(random.uniform(0.2, 0.7), 3)
            hr        = round(random.gauss(72, 5), 1)
            spo2      = round(random.gauss(98, 0.5), 1)
            valid     = 1

        else:                              # late night / winding down
            pir_count = random.randint(0, 4)
            pir_ratio = round(random.uniform(0.0, 0.2), 3)
            no_motion = round(random.uniform(60, 250), 1)
            app_on    = random.choice([0, 0, 0, 1])
            app_on_s  = round(random.uniform(0, 80), 1) if app_on else 0.0
            cur_mean  = round(random.uniform(0.0, 0.3), 3)
            hr        = round(random.gauss(62, 4), 1)
            spo2      = round(random.gauss(96.5, 0.5), 1)
            valid     = random.choice([0, 1])

        # Clamp physiological values
        hr   = max(40.0, min(hr,   160.0))
        spo2 = max(80.0, min(spo2, 100.0))
        hr_std   = round(random.uniform(1.5, 5.0), 2)
        spo2_std = round(random.uniform(0.2, 1.0), 2)
        app_switches = random.randint(0, 3) if app_on else 0
        cur_std  = round(random.uniform(0.02, 0.12), 3) if cur_mean > 0.05 else 0.0

        tachy    = 1 if hr > 110  else 0
        brady    = 1 if (hr < 50 and hr > 0) else 0
        spo2_low = 1 if (spo2 < 92 and spo2 > 0) else 0

        # Random anomaly injection
        if random.random() < anomaly_chance:
            anomaly_type = random.choice(["inactivity", "tachycardia", "spo2"])
            if anomaly_type == "inactivity" and 7 <= hour <= 21:
                no_motion = round(random.uniform(4000, 7200), 1)
                pir_count = 0
                pir_ratio = 0.0
            elif anomaly_type == "tachycardia":
                hr    = round(random.uniform(118, 140), 1)
                tachy = 1
            elif anomaly_type == "spo2":
                spo2     = round(random.uniform(82, 90), 1)
                spo2_low = 1
                tachy    = 1

        vectors.append(make_vector(
            hour, ts=ts,
            pir_count=pir_count, pir_ratio=pir_ratio, no_motion_s=no_motion,
            cur_mean=cur_mean,   cur_std=cur_std,
            app_on=app_on,       app_on_s=app_on_s, app_switches=app_switches,
            hr_mean=hr,          hr_std=hr_std,
            spo2_mean=spo2,      spo2_std=spo2_std,
            tachy=tachy,         brady=brady,  spo2_low=spo2_low,
            vitals_valid=valid,
        ))

    return vectors


def bulk_days(n_days: int, base_url: str, fast: bool):
    """Seed N days of synthetic data into the DB."""
    print(f"\n{'='*60}")
    print(f"  BULK SEED: {n_days} day(s) = {n_days * 288} vectors")
    print(f"{'='*60}")

    total   = 0
    errors  = 0
    t_start = time.time()

    for day in range(n_days):
        vectors = _daily_pattern(day_offset=day)
        for i, v in enumerate(vectors):
            try:
                resp = _post(base_url, v)
                if resp.status_code in (200, 201):
                    total += 1
                else:
                    errors += 1
            except requests.exceptions.ConnectionError:
                print(f"\n  ERROR: Cannot reach {base_url}. Is the stack running?")
                sys.exit(1)

            if not fast:
                time.sleep(0.02)   # 20 ms between posts in non-fast mode

        elapsed = time.time() - t_start
        rate    = total / elapsed if elapsed > 0 else 0
        print(f"  Day {day+1}/{n_days} — {total} vectors sent  "
              f"({rate:.0f}/s)  errors={errors}")

    elapsed = time.time() - t_start
    print(f"\n  Done. {total} vectors seeded in {elapsed:.1f}s.")
    print(f"  Open http://127.0.0.1:5001 to see the dashboard populate.")
    if total >= 100:
        print(f"  Ready to train models:  "
              f"cd hardware/dementia-monitor-backend && "
              f"python train_models.py")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Axialert patient behaviour simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenario name to run, or 'all' to run every scenario sequentially",
    )
    parser.add_argument(
        "--bulk-days", "-b",
        type=int, metavar="N",
        help="Seed N days of synthetic data (288 vectors/day)",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip inter-vector delays (useful for CI or bulk seeding)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Ingestion server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name in SCENARIOS:
            fn = SCENARIOS[name]
            print(f"  {name:<26} {fn.__doc__.strip()}")
        return

    if args.bulk_days:
        bulk_days(args.bulk_days, args.url, args.fast)
        return

    if not args.scenario:
        parser.print_help()
        return

    # Verify server is reachable
    try:
        r = requests.get(f"{args.url}/health", timeout=3)
        r.raise_for_status()
        print(f"[simulator] Connected to {args.url}  ✓")
    except Exception as e:
        print(f"[simulator] ERROR: Cannot reach {args.url}/health — {e}")
        print("[simulator] Start the stack with: ./run_stack.sh")
        sys.exit(1)

    random.seed(42)

    if args.scenario == "all":
        for name, fn in SCENARIOS.items():
            fn(args.url, args.fast)
    else:
        SCENARIOS[args.scenario](args.url, args.fast)

    print("\n[simulator] Done. Check http://127.0.0.1:5001 for results.")


if __name__ == "__main__":
    main()
