#!/usr/bin/env python3
"""
thingsboard_bridge.py
Optional bridge: SQLite backend -> ThingsBoard telemetry API.

This does not change the core local pipeline. It only mirrors latest telemetry
from dementia_monitor.db to ThingsBoard when enabled via environment variables.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class BridgeConfig:
    db_path: str
    host: str
    access_token: str
    use_https: bool
    interval_s: int
    timeout_s: int

    @property
    def telemetry_url(self) -> str:
        scheme = "https" if self.use_https else "http"
        return f"{scheme}://{self.host}/api/v1/{self.access_token}/telemetry"


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> BridgeConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.getenv("TB_DB_PATH", os.path.join(base_dir, "dementia_monitor.db"))
    host = os.getenv("TB_HOST", "demo.thingsboard.io")
    access_token = os.getenv("TB_ACCESS_TOKEN", "").strip()
    use_https = env_bool("TB_USE_HTTPS", True)
    interval_s = int(os.getenv("TB_INTERVAL_SEC", "10"))
    timeout_s = int(os.getenv("TB_HTTP_TIMEOUT_SEC", "5"))

    if not access_token:
        raise ValueError("TB_ACCESS_TOKEN is required")

    return BridgeConfig(
        db_path=db_path,
        host=host,
        access_token=access_token,
        use_https=use_https,
        interval_s=max(2, interval_s),
        timeout_s=max(2, timeout_s),
    )


def get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_vector(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM feature_vectors ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_alert_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])


def build_payload(vector: dict[str, Any], alert_count: int) -> dict[str, Any]:
    return {
        "vector_id": vector.get("id"),
        "hour": vector.get("hour"),
        "pir_ratio": vector.get("pir_ratio"),
        "pir_count": vector.get("pir_count"),
        "cur_mean": vector.get("cur_mean"),
        "cur_std": vector.get("cur_std"),
        "app_on": vector.get("app_on"),
        "app_on_s": vector.get("app_on_s"),
        "hr_mean": vector.get("hr_mean"),
        "spo2_mean": vector.get("spo2_mean"),
        "vitals_valid": vector.get("vitals_valid"),
        "a_motion": vector.get("a_motion"),
        "a_adl": vector.get("a_adl"),
        "a_vitals": vector.get("a_vitals"),
        "a_global": vector.get("a_global"),
        "anomaly_label": vector.get("anomaly_label") or "PENDING",  # FIX #20: UNKNOWN→PENDING
        "alerts_total": alert_count,
    }


def post_telemetry(cfg: BridgeConfig, payload: dict[str, Any]) -> None:
    resp = requests.post(cfg.telemetry_url, json=payload, timeout=cfg.timeout_s)
    resp.raise_for_status()


def run_bridge(cfg: BridgeConfig, once: bool = False) -> None:
    print("[TB] ThingsBoard bridge starting")
    print(f"[TB] DB: {cfg.db_path}")
    print(f"[TB] Host: {cfg.host}")
    print(f"[TB] URL: {cfg.telemetry_url.rsplit('/', 1)[0]}/<token>/telemetry")

    last_vector_id = -1

    while True:
        try:
            with contextlib.closing(get_db(cfg.db_path)) as conn:  # FIX #15: ensures close()
                vector = get_latest_vector(conn)
                if vector is None:
                    print("[TB] No vectors yet; waiting...")
                else:
                    vector_id = int(vector["id"])
                    if vector_id != last_vector_id:
                        alerts_total = get_alert_count(conn)
                        payload = build_payload(vector, alerts_total)
                        post_telemetry(cfg, payload)
                        last_vector_id = vector_id
                        print(
                            f"[TB] Synced vector #{vector_id} | "
                            f"A_global={payload.get('a_global')} | "
                            f"Label={payload.get('anomaly_label')}"
                        )
            if once:
                return
            time.sleep(cfg.interval_s)
        except KeyboardInterrupt:
            print("\n[TB] Bridge stopped")
            return
        except Exception as e:
            print(f"[TB] Error: {e}")
            if once:
                raise
            time.sleep(cfg.interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge local telemetry to ThingsBoard")
    parser.add_argument("--once", action="store_true", help="Run one sync cycle and exit")
    args = parser.parse_args()

    cfg = load_config()
    run_bridge(cfg, once=args.once)


if __name__ == "__main__":
    main()
