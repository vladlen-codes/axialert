"""
dashboard.py — AxiAlert Dementia Monitor
Drag-and-drop a patient_data.json to replay readings through the AI pipeline.
Open http://localhost:5001
"""

import json, os, time
import requests as _req
from flask import Flask, render_template_string, jsonify, Response, request

app      = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INGEST   = "http://127.0.0.1:7777"

# ─── Proxy helpers ────────────────────────────────────────────────────────────

def _fetch_recent(n=100):
    try:
        r = _req.get(f"{INGEST}/status?n={n}", timeout=2)
        return r.json() if r.ok else []
    except Exception:
        return []

def _fetch_patient():
    try:
        r = _req.get(f"{INGEST}/patient", timeout=2)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def build_payload(source=None):
    rows = _fetch_recent(100)
    if source:
        rows = [r for r in rows if r.get("source") == source]
    total = len(rows)
    alert_count = sum(
        1 for r in rows
        if r.get("anomaly_label", "NORMAL") not in ("NORMAL", "UNSCORED", None)
    )
    est_hours = round(total * 5 / 60, 1)
    latest    = rows[0] if rows else None
    ts_asc    = list(reversed(rows))
    return {
        "stats":      {"total_vectors": total, "total_alerts": alert_count, "est_hours": est_hours},
        "latest":     latest,
        "timeseries": ts_asc,
        "patient":    _fetch_patient(),
    }

# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AxiAlert — Dementia Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#07101f; --surface:#0f1e33; --card:#142238;
  --border:#1e3250; --text:#dce8f5; --muted:#4a6785;
  --green:#10b981; --yellow:#f59e0b; --red:#ef4444;
  --blue:#3b82f6; --purple:#8b5cf6; --cyan:#06b6d4;
  --orange:#f97316;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-size:14px;line-height:1.5}
a{color:var(--cyan)}

/* ── Header ── */
header{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;
  padding:16px 28px;border-bottom:1px solid var(--border);
  background:var(--surface);position:sticky;top:0;z-index:200;
}
.logo{font-family:'Space Mono',monospace;font-size:13px;letter-spacing:.15em;color:var(--cyan);text-transform:uppercase}
.logo span{color:var(--muted)}
#patient-bar{display:flex;align-items:center;gap:14px;font-size:13px}
#patient-name{font-weight:700;color:var(--text)}
#patient-meta{color:var(--muted);font-size:11px}
#risk-tag{padding:2px 9px;border-radius:12px;font-size:10px;font-weight:700;font-family:'Space Mono',monospace;letter-spacing:.08em}
.risk-HIGH{background:#ef444422;color:var(--red);border:1px solid var(--red)}
.risk-MEDIUM{background:#f59e0b22;color:var(--yellow);border:1px solid var(--yellow)}
.risk-LOW{background:#10b98122;color:var(--green);border:1px solid var(--green)}
#hdr-right{display:flex;align-items:center;gap:10px}
#status-badge{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;
  letter-spacing:.1em;font-family:'Space Mono',monospace;text-transform:uppercase}
.badge-ok{background:#10b98122;color:var(--green);border:1px solid var(--green)}
.badge-warning{background:#f59e0b22;color:var(--yellow);border:1px solid var(--yellow)}
.badge-critical{background:#ef444422;color:var(--red);border:1px solid var(--red);animation:pulse 1s infinite}
#conn-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block;margin-right:5px;transition:background .3s}
#conn-dot.live{background:var(--green);box-shadow:0 0 6px var(--green)}
#last-upd{font-size:11px;color:var(--muted);font-family:'Space Mono',monospace}

/* ── Layout ── */
main{padding:24px 28px;max-width:1500px;margin:0 auto}
.section-title{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-family:'Space Mono',monospace;margin-bottom:12px}

/* ── Drop zone ── */
#drop-zone{
  border:2px dashed var(--border);border-radius:12px;padding:36px 24px;
  text-align:center;cursor:pointer;transition:border-color .2s,background .2s;
  margin-bottom:20px;background:var(--surface);
}
#drop-zone.drag-over{border-color:var(--cyan);background:#06b6d411}
#drop-zone.loaded{border-color:var(--green);background:#10b98108;border-style:solid}
#drop-icon{font-size:36px;margin-bottom:8px}
#drop-label{color:var(--muted);font-size:13px}
#drop-label strong{color:var(--cyan)}
#replay-bar{display:none;margin-top:14px}
#replay-track{width:100%;height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:6px}
#replay-fill{height:100%;width:0%;background:var(--cyan);border-radius:3px;transition:width .3s}
#replay-label{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}
#replay-speed{margin-left:10px;font-size:11px;color:var(--muted)}

/* ── Sensor panel ── */
#sensor-panel{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;
  margin-bottom:20px;
}
.sensor-card{
  background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 14px;
  display:flex;flex-direction:column;align-items:center;gap:8px;position:relative;overflow:hidden;
}
.sensor-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--blue))}
.sensor-name{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-family:'Space Mono',monospace}
.sensor-val{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;color:var(--text)}
.sensor-unit{font-size:10px;color:var(--muted);margin-top:-4px}
.sensor-viz{width:100%;height:28px;display:flex;align-items:center;justify-content:center}

/* PIR pulsing dot */
.pir-dot{
  width:22px;height:22px;border-radius:50%;background:var(--blue);
  transition:all .4s;
}
.pir-dot.active{
  background:var(--cyan);box-shadow:0 0 12px var(--cyan);
  animation:pir-pulse .8s ease-in-out infinite;
}
@keyframes pir-pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.35);opacity:.7}}

/* Heartbeat bar */
.hb-wrap{width:100%;display:flex;align-items:center;gap:3px;justify-content:center}
.hb-bar{width:4px;border-radius:2px;background:var(--green);transition:height .3s}

/* SpO2 arc */
.spo2-ring{position:relative;width:42px;height:42px}
.spo2-ring svg{transform:rotate(-90deg)}
.spo2-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-family:'Space Mono',monospace;font-size:9px;font-weight:700}

/* LED matrix */
.led-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.led-cell{width:10px;height:10px;border-radius:2px;background:var(--border);transition:background .2s,box-shadow .2s}
.led-cell.on{background:var(--red);box-shadow:0 0 6px var(--red)}

/* Buzzer */
.buzzer-icon{font-size:22px;transition:transform .1s}
.buzzer-icon.buz{animation:buzz .12s linear infinite}
@keyframes buzz{0%{transform:translateX(-2px)}50%{transform:translateX(2px)}100%{transform:translateX(-2px)}}

/* ── Stats grid ── */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;margin-bottom:20px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;
  position:relative;overflow:hidden;transition:border-color .3s}
.stat-card.flash{border-color:var(--cyan)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--blue))}
.stat-label{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  font-family:'Space Mono',monospace;margin-bottom:5px}
.stat-value{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;line-height:1}
.stat-unit{font-size:10px;color:var(--muted);margin-top:2px}

/* ── Charts ── */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.chart-full{grid-column:1/-1}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px}
.chart-wrap{position:relative;height:170px}
.chart-wrap-tall{position:relative;height:210px}

/* ── Sub-scores ── */
.subscores{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.subscore-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center}
.subscore-bar-wrap{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin:8px 0 4px}
.subscore-bar{height:100%;border-radius:3px;transition:width .4s,background .4s}
.subscore-label{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-family:'Space Mono',monospace}
.subscore-val{font-family:'Space Mono',monospace;font-size:18px;font-weight:700;margin-top:4px}

/* ── Alert log ── */
.alert-log{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:14px}
.alert-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px}
.alert-row:last-child{border-bottom:none}
.alert-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-critical{background:var(--red)} .dot-warning{background:var(--yellow)}
.alert-time{color:var(--muted);font-family:'Space Mono',monospace;font-size:10px;min-width:130px}
.alert-type{font-weight:600}
.alert-score{margin-left:auto;font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}
.alert-detail{font-size:11px;color:var(--muted)}
.empty-state{color:var(--muted);text-align:center;padding:20px;font-size:13px}

/* ── Patient profile panel ── */
#profile-panel{
  background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;
  margin-bottom:20px;display:none;
}
.profile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.profile-field{display:flex;flex-direction:column;gap:2px}
.profile-key{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-family:'Space Mono',monospace}
.profile-val{font-size:13px;font-weight:500}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
@media(max-width:700px){.chart-grid{grid-template-columns:1fr}.subscores{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <div class="logo">Axi<span>Alert</span> <span style="font-size:10px">// dementia monitor</span></div>
  <div id="patient-bar">
    <span id="patient-name" style="display:none"></span>
    <span id="patient-meta"></span>
    <span id="risk-tag" style="display:none"></span>
  </div>
  <div id="hdr-right">
    <span id="status-badge" class="badge-ok">OK</span>
    <span id="last-upd"><span id="conn-dot"></span>Connecting…</span>
  </div>
</header>

<main>

  <!-- Mode toggle -->
  <div style="display:flex;align-items:center;gap:0;margin-bottom:20px;background:var(--surface);
              border:1px solid var(--border);border-radius:10px;overflow:hidden;width:fit-content">
    <button id="btn-live" onclick="setMode('live')"
      style="padding:10px 28px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
             letter-spacing:.1em;border:none;cursor:pointer;transition:all .2s;
             background:var(--cyan);color:#07131f">
      ⚡ LIVE SENSORS
    </button>
    <button id="btn-replay" onclick="setMode('replay')"
      style="padding:10px 28px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
             letter-spacing:.1em;border:none;cursor:pointer;transition:all .2s;
             background:var(--surface);color:var(--muted)">
      📂 REPLAY FILE
    </button>
  </div>

  <!-- Hardware connection panel -->
  <div id="hw-panel" style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:20px">
    <div class="section-title">ESP32 Hardware Connection</div>
    <div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:10px">
      <div id="hw-dot" style="width:10px;height:10px;border-radius:50%;background:var(--muted);flex-shrink:0"></div>
      <span id="hw-status" style="font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)">Not configured</span>
      <input id="esp32-ip" type="text" placeholder="ESP32 IP  e.g. 192.168.1.55"
        style="flex:1;min-width:180px;background:var(--bg);border:1px solid var(--border);border-radius:6px;
               padding:6px 10px;font-family:'Space Mono',monospace;font-size:11px;color:var(--text)">
      <button onclick="setEsp32()" style="background:var(--cyan);color:#07131f;border:none;border-radius:6px;
        padding:7px 14px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;cursor:pointer">
        CONNECT
      </button>
      <button onclick="pingEsp32()" style="background:var(--border);color:var(--text);border:none;border-radius:6px;
        padding:7px 14px;font-family:'Space Mono',monospace;font-size:11px;cursor:pointer">
        PING
      </button>
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-top:10px">
      <button onclick="silenceAlert()"
        style="background:#ef444422;color:var(--red);border:1px solid var(--red);border-radius:6px;
               padding:7px 18px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
               cursor:pointer;letter-spacing:.08em">
        🔕 SILENCE ALERT
      </button>
      <span id="silence-msg" style="font-family:'Space Mono',monospace;font-size:10px;color:var(--muted)"></span>
    </div>
    <div id="hw-detail" style="font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);margin-top:8px">
      Flash the ESP32 with PlatformIO, open Serial Monitor to get its IP, then paste it above.
    </div>
  </div>

  <!-- Drop zone -->
  <div id="drop-zone" onclick="document.getElementById('file-in').click()">
    <input type="file" id="file-in" accept=".json" style="display:none" onchange="handleFile(this.files[0])">
    <div id="drop-icon">📂</div>
    <div id="drop-label">Drag &amp; drop <strong>patient_data.json</strong> here — or click to browse</div>
    <div id="replay-bar">
      <div id="replay-track"><div id="replay-fill"></div></div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px">
        <span id="replay-label">Replaying…</span>
        <div style="display:flex;align-items:center;gap:8px">
          <span id="replay-speed"></span>
          <button id="replay-stop" onclick="stopReplay()"
            style="background:var(--red);color:#fff;border:none;border-radius:5px;
                   padding:3px 10px;font-family:'Space Mono',monospace;font-size:10px;
                   cursor:pointer;font-weight:700">STOP</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Patient profile -->
  <div id="profile-panel">
    <div class="section-title">Patient Profile</div>
    <div class="profile-grid" id="profile-grid"></div>
  </div>

  <!-- Sensor panel -->
  <div class="section-title">Live Sensor State</div>
  <div id="sensor-panel">

    <div class="sensor-card" style="--accent:var(--blue)">
      <div class="sensor-name">PIR Motion</div>
      <div class="sensor-viz"><div class="pir-dot" id="pir-dot"></div></div>
      <div class="sensor-val" id="sv-pir">—</div>
      <div class="sensor-unit">activity ratio</div>
    </div>

    <div class="sensor-card" style="--accent:var(--green)">
      <div class="sensor-name">Heart Rate</div>
      <div class="sensor-viz hb-wrap" id="hb-bars">
        <div class="hb-bar" style="height:6px"></div>
        <div class="hb-bar" style="height:14px"></div>
        <div class="hb-bar" style="height:22px"></div>
        <div class="hb-bar" style="height:14px"></div>
        <div class="hb-bar" style="height:6px"></div>
      </div>
      <div class="sensor-val" id="sv-hr">—</div>
      <div class="sensor-unit">bpm</div>
    </div>

    <div class="sensor-card" style="--accent:var(--cyan)">
      <div class="sensor-name">SpO₂</div>
      <div class="sensor-viz">
        <div class="spo2-ring">
          <svg width="42" height="42" viewBox="0 0 42 42">
            <circle cx="21" cy="21" r="17" fill="none" stroke="#1e3250" stroke-width="4"/>
            <circle id="spo2-arc" cx="21" cy="21" r="17" fill="none" stroke="#06b6d4"
                    stroke-width="4" stroke-dasharray="107" stroke-dashoffset="107"
                    stroke-linecap="round" style="transition:stroke-dashoffset .5s"/>
          </svg>
          <div class="spo2-val" id="sv-spo2-sm">—%</div>
        </div>
      </div>
      <div class="sensor-val" id="sv-spo2">—</div>
      <div class="sensor-unit">% saturation</div>
    </div>

    <div class="sensor-card" style="--accent:var(--purple)">
      <div class="sensor-name">Current (ACS712)</div>
      <div class="sensor-viz" style="flex-direction:column;gap:4px;width:100%">
        <div style="width:100%;height:8px;background:var(--border);border-radius:4px;overflow:hidden">
          <div id="cur-bar" style="height:100%;width:0%;background:var(--purple);border-radius:4px;transition:width .4s"></div>
        </div>
        <div style="font-size:9px;color:var(--muted);font-family:'Space Mono',monospace" id="app-state">APPLIANCE OFF</div>
      </div>
      <div class="sensor-val" id="sv-cur">—</div>
      <div class="sensor-unit">A mean</div>
    </div>

    <div class="sensor-card" style="--accent:var(--yellow)">
      <div class="sensor-name">No Motion</div>
      <div class="sensor-viz">
        <div style="font-size:22px" id="motion-icon">🚶</div>
      </div>
      <div class="sensor-val" id="sv-nomotion">—</div>
      <div class="sensor-unit">seconds still</div>
    </div>

    <div class="sensor-card" style="--accent:var(--red)">
      <div class="sensor-name">LED Matrix</div>
      <div class="sensor-viz">
        <div class="led-grid" id="led-grid">
          <div class="led-cell"></div><div class="led-cell"></div>
          <div class="led-cell"></div><div class="led-cell"></div>
          <div class="led-cell"></div><div class="led-cell"></div>
          <div class="led-cell"></div><div class="led-cell"></div>
        </div>
      </div>
      <div class="sensor-val" id="sv-led" style="font-size:13px">OFF</div>
      <div class="sensor-unit">alert indicator</div>
    </div>

    <div class="sensor-card" style="--accent:var(--orange)">
      <div class="sensor-name">Buzzer</div>
      <div class="sensor-viz"><div class="buzzer-icon" id="buzzer-icon">🔇</div></div>
      <div class="sensor-val" id="sv-buzzer" style="font-size:13px">SILENT</div>
      <div class="sensor-unit">audio alert</div>
    </div>

    <div class="sensor-card" style="--accent:var(--cyan)">
      <div class="sensor-name">Hour of Day</div>
      <div class="sensor-viz">
        <div style="font-size:22px" id="hour-icon">🕐</div>
      </div>
      <div class="sensor-val" id="sv-hour">—</div>
      <div class="sensor-unit">24h clock</div>
    </div>

  </div>

  <!-- Sub-scores -->
  <div class="section-title">AI Sub-scores</div>
  <div class="subscores">
    <div class="subscore-card">
      <div class="subscore-label">Motion Score</div>
      <div class="subscore-bar-wrap"><div class="subscore-bar" id="bar-motion" style="width:0%;background:var(--blue)"></div></div>
      <div class="subscore-val" id="sc-motion" style="color:var(--blue)">—</div>
    </div>
    <div class="subscore-card">
      <div class="subscore-label">ADL Score</div>
      <div class="subscore-bar-wrap"><div class="subscore-bar" id="bar-adl" style="width:0%;background:var(--purple)"></div></div>
      <div class="subscore-val" id="sc-adl" style="color:var(--purple)">—</div>
    </div>
    <div class="subscore-card">
      <div class="subscore-label">Vitals Score</div>
      <div class="subscore-bar-wrap"><div class="subscore-bar" id="bar-vitals" style="width:0%;background:var(--cyan)"></div></div>
      <div class="subscore-val" id="sc-vitals" style="color:var(--cyan)">—</div>
    </div>
  </div>

  <!-- Stats grid -->
  <div class="stat-grid">
    <div class="stat-card" style="--accent:var(--cyan)">
      <div class="stat-label">Readings</div>
      <div class="stat-value" id="s-total">—</div>
      <div class="stat-unit">processed</div>
    </div>
    <div class="stat-card" style="--accent:var(--blue)">
      <div class="stat-label">Data Span</div>
      <div class="stat-value" id="s-hours">—</div>
      <div class="stat-unit">hours est.</div>
    </div>
    <div class="stat-card" style="--accent:var(--green)">
      <div class="stat-label">Heart Rate</div>
      <div class="stat-value" id="s-hr">—</div>
      <div class="stat-unit">bpm latest</div>
    </div>
    <div class="stat-card" style="--accent:var(--cyan)">
      <div class="stat-label">SpO₂</div>
      <div class="stat-value" id="s-spo2">—</div>
      <div class="stat-unit">% latest</div>
    </div>
    <div class="stat-card" style="--accent:var(--yellow)">
      <div class="stat-label">A_global</div>
      <div class="stat-value" id="s-score">—</div>
      <div class="stat-unit">anomaly score</div>
    </div>
    <div class="stat-card" style="--accent:var(--red)">
      <div class="stat-label">Alerts</div>
      <div class="stat-value" id="s-alerts">—</div>
      <div class="stat-unit">total triggered</div>
    </div>
    <div class="stat-card" style="--accent:var(--purple)">
      <div class="stat-label">Last Motion</div>
      <div class="stat-value" id="s-nomotion">—</div>
      <div class="stat-unit">seconds ago</div>
    </div>
    <div class="stat-card" style="--accent:var(--orange)">
      <div class="stat-label">Appliance</div>
      <div class="stat-value" id="s-appliance">—</div>
      <div class="stat-unit">on duration (s)</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="chart-grid">
    <div class="chart-card chart-full">
      <div class="section-title">Anomaly Score Timeline — A_global</div>
      <div class="chart-wrap-tall"><canvas id="c-anomaly"></canvas></div>
    </div>
  </div>
  <div class="chart-grid">
    <div class="chart-card">
      <div class="section-title">Sub-scores Over Time</div>
      <div class="chart-wrap"><canvas id="c-subscores"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="section-title">Heart Rate (bpm)</div>
      <div class="chart-wrap"><canvas id="c-hr"></canvas></div>
    </div>
  </div>
  <div class="chart-grid">
    <div class="chart-card">
      <div class="section-title">PIR Activity Ratio</div>
      <div class="chart-wrap"><canvas id="c-motion"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="section-title">SpO₂ (%)</div>
      <div class="chart-wrap"><canvas id="c-spo2"></canvas></div>
    </div>
  </div>
  <div class="chart-grid">
    <div class="chart-card chart-full">
      <div class="section-title">Appliance Usage — ON Duration (s)</div>
      <div class="chart-wrap"><canvas id="c-appliance"></canvas></div>
    </div>
  </div>

  <!-- Alert log -->
  <div class="alert-log">
    <div class="section-title">Anomaly Event Log</div>
    <div id="alert-log-body"><div class="empty-state">No anomalies yet — waiting for data</div></div>
  </div>

</main>

<script>
// ── Chart setup ──────────────────────────────────────────────────────────────
let charts = {};

function makeChart(id, label, color, yMin, yMax) {
  const ctx = document.getElementById(id).getContext('2d');
  charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels:[], datasets:[{label, data:[], borderColor:color,
      backgroundColor:color+'22', borderWidth:1.5, pointRadius:2, fill:true, tension:0.3}] },
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:300},
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#4a6785',font:{family:'Space Mono',size:9},maxTicksLimit:8},grid:{color:'#1e3250'}},
        y:{min:yMin,max:yMax,ticks:{color:'#4a6785',font:{family:'Space Mono',size:9}},grid:{color:'#1e3250'}}
      }
    }
  });
}

function makeAnomalyChart() {
  const ctx = document.getElementById('c-anomaly').getContext('2d');
  charts['c-anomaly'] = new Chart(ctx, {
    type:'line',
    data:{labels:[],datasets:[
      {label:'A_global',data:[],borderColor:'#06b6d4',backgroundColor:'#06b6d422',borderWidth:2,pointRadius:3,fill:true,tension:0.3},
      {label:'Warning', data:[],borderColor:'#f59e0b',borderDash:[5,5],borderWidth:1,pointRadius:0,fill:false},
      {label:'Critical',data:[],borderColor:'#ef4444',borderDash:[5,5],borderWidth:1,pointRadius:0,fill:false},
    ]},
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:300},
      plugins:{legend:{labels:{color:'#4a6785',font:{family:'Space Mono',size:10},boxWidth:18}}},
      scales:{
        x:{ticks:{color:'#4a6785',font:{family:'Space Mono',size:9},maxTicksLimit:10},grid:{color:'#1e3250'}},
        y:{min:0,max:1,ticks:{color:'#4a6785',font:{family:'Space Mono',size:9}},grid:{color:'#1e3250'}}
      }
    }
  });
}

function makeSubscoreChart() {
  const ctx = document.getElementById('c-subscores').getContext('2d');
  charts['c-subscores'] = new Chart(ctx, {
    type:'line',
    data:{labels:[],datasets:[
      {label:'Motion', data:[],borderColor:'#3b82f6',backgroundColor:'transparent',borderWidth:1.5,pointRadius:2,tension:0.3},
      {label:'ADL',    data:[],borderColor:'#8b5cf6',backgroundColor:'transparent',borderWidth:1.5,pointRadius:2,tension:0.3},
      {label:'Vitals', data:[],borderColor:'#06b6d4',backgroundColor:'transparent',borderWidth:1.5,pointRadius:2,tension:0.3},
    ]},
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:300},
      plugins:{legend:{labels:{color:'#4a6785',font:{family:'Space Mono',size:10},boxWidth:16}}},
      scales:{
        x:{ticks:{color:'#4a6785',font:{family:'Space Mono',size:9},maxTicksLimit:8},grid:{color:'#1e3250'}},
        y:{min:0,max:1,ticks:{color:'#4a6785',font:{family:'Space Mono',size:9}},grid:{color:'#1e3250'}}
      }
    }
  });
}

makeAnomalyChart();
makeSubscoreChart();
makeChart('c-hr',        'Heart Rate',   '#10b981', 40,  140);
makeChart('c-motion',    'PIR Ratio',    '#3b82f6',  0,    1);
makeChart('c-spo2',      'SpO2',         '#06b6d4', 85,  100);
makeChart('c-appliance', 'ON seconds',   '#8b5cf6',  0, null);

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmtTime(iso) { return new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}); }
function fmtDT(iso) {
  const d=new Date(iso);
  return d.toLocaleDateString([],{month:'short',day:'numeric'})+' '+
         d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
}
function scoreColor(v) {
  if (v == null) return '#4a6785';
  if (v >= 0.80) return '#ef4444';
  if (v >= 0.65) return '#f59e0b';
  return '#10b981';
}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}

// ── Sensor panel ─────────────────────────────────────────────────────────────
let _ledBlink = null;

function updateSensors(r) {
  if (!r) return;

  // PIR
  const pir = r.pir_ratio ?? 0;
  document.getElementById('sv-pir').textContent = pir.toFixed(3);
  const pirDot = document.getElementById('pir-dot');
  if (pir > 0.05) pirDot.classList.add('active');
  else            pirDot.classList.remove('active');

  // Heart rate bars
  const hr = r.hr_mean ?? 0;
  document.getElementById('sv-hr').textContent = hr ? hr.toFixed(0) : '—';
  const heights = hr > 0 ? [6,10,18,26,18,10,6] : [4,4,4,4,4,4,4];
  const bars = document.querySelectorAll('.hb-bar');
  bars.forEach((b,i) => { b.style.height = (heights[i % heights.length] || 6) + 'px'; });

  // SpO2
  const spo2 = r.spo2_mean ?? 0;
  document.getElementById('sv-spo2').textContent = spo2 ? spo2.toFixed(1) : '—';
  document.getElementById('sv-spo2-sm').textContent = spo2 ? spo2.toFixed(0)+'%' : '—';
  const circ = 2 * Math.PI * 17;  // ~107
  const pct = spo2 ? clamp((spo2 - 85) / 15, 0, 1) : 0;
  document.getElementById('spo2-arc').setAttribute('stroke-dashoffset', (circ * (1 - pct)).toFixed(1));
  document.getElementById('spo2-arc').style.stroke = spo2 >= 92 ? '#06b6d4' : '#ef4444';

  // Current / appliance
  const cur = r.cur_mean ?? 0;
  document.getElementById('sv-cur').textContent = cur ? cur.toFixed(3) : '—';
  document.getElementById('cur-bar').style.width = clamp(cur * 100, 0, 100) + '%';
  document.getElementById('app-state').textContent = r.app_on ? 'APPLIANCE ON' : 'APPLIANCE OFF';

  // No motion
  const nm = r.no_motion_s ?? -1;
  document.getElementById('sv-nomotion').textContent = nm >= 0 ? nm.toFixed(0) : '—';
  document.getElementById('motion-icon').textContent = (nm > 3600) ? '🛋️' : (pir > 0.05) ? '🚶' : '💤';

  // Hour
  const h = r.hour ?? null;
  document.getElementById('sv-hour').textContent = h != null ? String(h).padStart(2,'0')+':00' : '—';
  const icons = ['🌙','🌙','🌙','🌙','🌙','🌅','🌅','☀️','☀️','☀️','☀️','☀️',
                 '☀️','☀️','☀️','☀️','☀️','🌆','🌆','🌆','🌙','🌙','🌙','🌙'];
  document.getElementById('hour-icon').textContent = h != null ? icons[h] : '🕐';

  // LED + Buzzer — based on anomaly
  const lbl = r.anomaly_label || 'NORMAL';
  const isAlert = lbl.startsWith('CRITICAL') || lbl.startsWith('WARNING');
  const isCrit  = lbl.startsWith('CRITICAL');

  if (_ledBlink) { clearInterval(_ledBlink); _ledBlink = null; }
  const ledCells = document.querySelectorAll('.led-cell');
  if (isAlert) {
    let on = false;
    _ledBlink = setInterval(() => {
      on = !on;
      ledCells.forEach(c => on ? c.classList.add('on') : c.classList.remove('on'));
    }, isCrit ? 300 : 500);
    document.getElementById('sv-led').textContent = isCrit ? 'BLINK FAST' : 'BLINK SLOW';
    document.getElementById('sv-led').style.color = isCrit ? '#ef4444' : '#f59e0b';
    document.getElementById('buzzer-icon').textContent = '🔊';
    document.getElementById('buzzer-icon').classList.add('buz');
    document.getElementById('sv-buzzer').textContent = isCrit ? 'CRITICAL' : 'WARNING';
    document.getElementById('sv-buzzer').style.color = isCrit ? '#ef4444' : '#f59e0b';
  } else {
    ledCells.forEach(c => c.classList.remove('on'));
    document.getElementById('sv-led').textContent = 'OFF';
    document.getElementById('sv-led').style.color = '#4a6785';
    document.getElementById('buzzer-icon').textContent = '🔇';
    document.getElementById('buzzer-icon').classList.remove('buz');
    document.getElementById('sv-buzzer').textContent = 'SILENT';
    document.getElementById('sv-buzzer').style.color = '#4a6785';
  }

  // Sub-scores
  const am = r.a_motion, aa = r.a_adl, av = r.a_vitals;
  function setSubscore(id, barId, val) {
    const el = document.getElementById(id);
    const bar = document.getElementById(barId);
    if (val != null) {
      el.textContent = val.toFixed(3);
      el.style.color = scoreColor(val);
      bar.style.width = (val * 100).toFixed(1) + '%';
      bar.style.background = scoreColor(val);
    } else {
      el.textContent = '—';
    }
  }
  setSubscore('sc-motion','bar-motion', am);
  setSubscore('sc-adl',   'bar-adl',   aa);
  setSubscore('sc-vitals','bar-vitals', av);
}

// ── Main data apply ──────────────────────────────────────────────────────────
function applyData(d) {
  const stats  = d.stats  || {};
  const latest = d.latest || null;
  const ts     = d.timeseries || [];
  const n      = ts.length;

  // Stats
  document.getElementById('s-total').textContent   = stats.total_vectors ?? '—';
  document.getElementById('s-hours').textContent   = stats.est_hours     ?? '—';
  document.getElementById('s-alerts').textContent  = stats.total_alerts  ?? '—';

  if (latest) {
    document.getElementById('s-hr').textContent       = latest.hr_mean   ? latest.hr_mean.toFixed(0)   : '—';
    document.getElementById('s-spo2').textContent     = latest.spo2_mean ? latest.spo2_mean.toFixed(1) : '—';
    document.getElementById('s-score').textContent    = latest.a_global  != null ? latest.a_global.toFixed(3) : '—';
    document.getElementById('s-nomotion').textContent = latest.no_motion_s != null ? latest.no_motion_s.toFixed(0) : '—';
    document.getElementById('s-appliance').textContent= latest.app_on_s != null ? latest.app_on_s.toFixed(0) : '—';

    // Status badge
    const badge = document.getElementById('status-badge');
    const lbl   = latest.anomaly_label || '';
    if (lbl.startsWith('CRITICAL'))     { badge.className='badge-critical'; badge.textContent='CRITICAL'; }
    else if (lbl.startsWith('WARNING')) { badge.className='badge-warning';  badge.textContent='WARNING'; }
    else                                { badge.className='badge-ok';       badge.textContent='OK'; }

    updateSensors(latest);
  }

  // Patient info
  if (d.patient && d.patient.name) applyPatient(d.patient);

  // Charts
  const labels = ts.map(r => fmtTime(r.received_at));

  if (charts['c-anomaly']) {
    charts['c-anomaly'].data.labels = labels;
    charts['c-anomaly'].data.datasets[0].data = ts.map(r => r.a_global ?? null);
    charts['c-anomaly'].data.datasets[1].data = Array(n).fill(0.65);
    charts['c-anomaly'].data.datasets[2].data = Array(n).fill(0.80);
    // colour individual points by severity
    charts['c-anomaly'].data.datasets[0].pointBackgroundColor = ts.map(r => scoreColor(r.a_global));
    charts['c-anomaly'].update('none');
  }
  if (charts['c-subscores']) {
    charts['c-subscores'].data.labels = labels;
    charts['c-subscores'].data.datasets[0].data = ts.map(r => r.a_motion ?? null);
    charts['c-subscores'].data.datasets[1].data = ts.map(r => r.a_adl    ?? null);
    charts['c-subscores'].data.datasets[2].data = ts.map(r => r.a_vitals ?? null);
    charts['c-subscores'].update('none');
  }
  if (charts['c-hr'])        { charts['c-hr'].data.labels = labels;        charts['c-hr'].data.datasets[0].data        = ts.map(r => r.hr_mean    || null); charts['c-hr'].update('none'); }
  if (charts['c-motion'])    { charts['c-motion'].data.labels = labels;    charts['c-motion'].data.datasets[0].data    = ts.map(r => r.pir_ratio);          charts['c-motion'].update('none'); }
  if (charts['c-spo2'])      { charts['c-spo2'].data.labels = labels;      charts['c-spo2'].data.datasets[0].data      = ts.map(r => r.spo2_mean  || null); charts['c-spo2'].update('none'); }
  if (charts['c-appliance']) { charts['c-appliance'].data.labels = labels; charts['c-appliance'].data.datasets[0].data = ts.map(r => r.app_on_s);           charts['c-appliance'].update('none'); }

  // Alert log
  const logEl  = document.getElementById('alert-log-body');
  const alerts = ts.filter(r => r.anomaly_label && !['NORMAL','UNSCORED'].includes(r.anomaly_label))
                   .slice(-20).reverse();
  if (alerts.length === 0) {
    logEl.innerHTML = '<div class="empty-state">No anomalies detected yet</div>';
  } else {
    logEl.innerHTML = alerts.map(r => {
      const isCrit = r.anomaly_label.startsWith('CRITICAL');
      const col    = isCrit ? '#ef4444' : '#f59e0b';
      const score  = r.a_global != null ? r.a_global.toFixed(3) : '—';
      const detail = `HR ${r.hr_mean ? r.hr_mean.toFixed(0) : '—'}bpm · SpO₂ ${r.spo2_mean ? r.spo2_mean.toFixed(1) : '—'}% · PIR ${r.pir_ratio?.toFixed(2) ?? '—'}`;
      return `<div class="alert-row">
        <div class="alert-dot dot-${isCrit?'critical':'warning'}"></div>
        <span class="alert-time">${fmtDT(r.received_at)}</span>
        <div style="flex:1;min-width:0">
          <div class="alert-type" style="color:${col}">${r.anomaly_label}</div>
          <div class="alert-detail">${detail}</div>
        </div>
        <span class="alert-score">A=${score}</span>
      </div>`;
    }).join('');
  }

  document.getElementById('last-upd').innerHTML =
    '<span id="conn-dot" class="live"></span>Live · ' + new Date().toLocaleTimeString();
}

// ── Patient profile ──────────────────────────────────────────────────────────
function applyPatient(p) {
  if (!p || !p.name) return;
  document.getElementById('patient-name').textContent = p.name;
  document.getElementById('patient-name').style.display = 'inline';

  const meta = [p.age ? p.age + ' yrs' : null, p.condition, p.ward].filter(Boolean).join(' · ');
  document.getElementById('patient-meta').textContent = meta;

  const risk = p.risk_level || '';
  const rtag = document.getElementById('risk-tag');
  rtag.textContent = risk + ' RISK';
  rtag.className = 'risk-' + risk;
  rtag.style.display = 'inline';

  const panel  = document.getElementById('profile-panel');
  const grid   = document.getElementById('profile-grid');
  const fields = [
    ['Patient ID',  p.id],
    ['Age',         p.age ? p.age + ' years' : null],
    ['Condition',   p.condition],
    ['Ward',        p.ward],
    ['Caregiver',   p.caregiver],
    ['Device',      p.device_id || (window._sessionInfo && window._sessionInfo.device_id)],
    ['Location',    p.location  || (window._sessionInfo && window._sessionInfo.location)],
    ['Baseline HR', p.baseline_hr ? p.baseline_hr + ' bpm' : null],
    ['Baseline SpO₂', p.baseline_spo2 ? p.baseline_spo2 + '%' : null],
    ['Risk Level',  p.risk_level],
    ['Emergency',   p.emergency_contact],
  ].filter(([,v]) => v != null);

  grid.innerHTML = fields.map(([k,v]) => `
    <div class="profile-field">
      <div class="profile-key">${k}</div>
      <div class="profile-val">${v}</div>
    </div>`).join('');
  panel.style.display = 'block';
}

// ── SSE state (declared early — setMode calls connectSSE on init) ─────────────
let _sse = null;

// ── Mode toggle ──────────────────────────────────────────────────────────────
let _mode = 'live';

function setMode(mode) {
  _mode = mode;
  const live   = document.getElementById('btn-live');
  const replay = document.getElementById('btn-replay');
  const dz     = document.getElementById('drop-zone');

  if (mode === 'live') {
    live.style.background   = 'var(--cyan)';
    live.style.color        = '#07131f';
    replay.style.background = 'var(--surface)';
    replay.style.color      = 'var(--muted)';
    dz.style.display        = 'none';
    document.getElementById('hw-panel').style.display = 'block';
  } else {
    replay.style.background = 'var(--cyan)';
    replay.style.color      = '#07131f';
    live.style.background   = 'var(--surface)';
    live.style.color        = 'var(--muted)';
    dz.style.display        = 'block';
    document.getElementById('hw-panel').style.display = 'none';
  }
  connectSSE();
}

// Start in live mode — hide drop zone
setMode('live');

// ── Hardware connection ───────────────────────────────────────────────────────
async function silenceAlert() {
  const msg = document.getElementById('silence-msg');
  msg.style.color = 'var(--yellow)';
  msg.textContent = 'Sending…';
  try {
    const r = await fetch('/api/esp32/silence', { method: 'POST' });
    const d = await r.json();
    if (d.ok) {
      msg.style.color = 'var(--green)';
      msg.textContent = 'Alert silenced ✓';
    } else {
      msg.style.color = 'var(--red)';
      msg.textContent = d.error || 'Failed';
    }
  } catch(e) {
    msg.style.color = 'var(--red)';
    msg.textContent = 'Request failed';
  }
  setTimeout(() => { msg.textContent = ''; }, 4000);
}

async function setEsp32() {
  const ip = document.getElementById('esp32-ip').value.trim();
  if (!ip) { alert('Enter the ESP32 IP address first.'); return; }
  try {
    const r = await fetch('/api/esp32/set', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ip}),
    });
    const d = await r.json();
    if (d.ok) {
      document.getElementById('hw-detail').textContent = `Target set → ${d.ip}:${d.port}  — pinging…`;
      await pingEsp32();
    } else {
      document.getElementById('hw-detail').textContent = 'Error: ' + (d.error || 'unknown');
    }
  } catch(e) {
    document.getElementById('hw-detail').textContent = 'Request failed: ' + e;
  }
}

async function pingEsp32() {
  const dot    = document.getElementById('hw-dot');
  const status = document.getElementById('hw-status');
  const detail = document.getElementById('hw-detail');
  status.textContent = 'Pinging…';
  dot.style.background = 'var(--yellow)';
  try {
    const r = await fetch('/api/esp32/ping');
    const d = await r.json();
    if (d.reachable) {
      dot.style.background = 'var(--green)';
      dot.style.boxShadow  = '0 0 6px var(--green)';
      status.style.color   = 'var(--green)';
      status.textContent   = `CONNECTED  ${d.ip}:${d.port}`;
      detail.textContent   = `ESP32 is live — alerts will now trigger the physical LED and buzzer.`;
      document.getElementById('esp32-ip').value = d.ip;
    } else {
      dot.style.background = 'var(--red)';
      dot.style.boxShadow  = '0 0 6px var(--red)';
      status.style.color   = 'var(--red)';
      status.textContent   = `UNREACHABLE  ${d.ip || ''}`;
      detail.textContent   = d.reason || 'Cannot reach ESP32. Check IP, WiFi, and that firmware is running.';
    }
  } catch(e) {
    dot.style.background = 'var(--muted)';
    status.textContent   = 'Ping failed';
    detail.textContent   = String(e);
  }
}

// Auto-check hardware status on load
(async () => {
  try {
    const r = await fetch('/api/esp32/ping');
    const d = await r.json();
    if (d.ip) {
      document.getElementById('esp32-ip').value = d.ip;
      if (d.reachable) pingEsp32();
      else {
        document.getElementById('hw-status').textContent = `Configured: ${d.ip} (not reachable)`;
        document.getElementById('hw-detail').textContent = 'ESP32 IP is set but not responding. Check it is powered and on WiFi.';
      }
    }
  } catch(_) {}
})();

// ── SSE live stream ──────────────────────────────────────────────────────────
function connectSSE() {
  if (_sse) { _sse.close(); _sse = null; }
  const url = '/stream?source=' + _mode;
  _sse = new EventSource(url);
  _sse.addEventListener('init',   e => applyData(JSON.parse(e.data)));
  _sse.addEventListener('update', e => {
    applyData(JSON.parse(e.data));
    ['s-total','s-hr','s-spo2','s-score','s-alerts'].forEach(id => {
      const el = document.getElementById(id)?.closest('.stat-card');
      if (el) { el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 600); }
    });
  });
  _sse.onerror = () => {
    document.getElementById('last-upd').innerHTML = '<span id="conn-dot"></span>Reconnecting…';
    _sse.close(); _sse = null;
    setTimeout(connectSSE, 3000);
  };
}

// ── Drag & drop ──────────────────────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

async function handleFile(file) {
  if (!file || !file.name.endsWith('.json')) {
    alert('Please drop a .json file.');
    return;
  }
  let body;
  try { body = JSON.parse(await file.text()); }
  catch(e) { alert('Invalid JSON: ' + e); return; }

  const patient  = body.patient  || {};
  const readings = body.readings || [];
  const session  = body.session  || {};
  window._sessionInfo = { ...session, ...patient };

  if (!readings.length) { alert('No readings found in file.'); return; }

  // Show patient info immediately
  applyPatient({...patient, ...session});

  // Show drop zone as loaded
  dz.classList.add('loaded');
  document.getElementById('drop-icon').textContent = '✅';
  document.getElementById('drop-label').innerHTML =
    `<strong>${file.name}</strong> — ${readings.length} readings loaded. Replaying…`;
  document.getElementById('replay-bar').style.display = 'block';

  // Replay
  await replayReadings(readings);
}

let _replaySpeed = 400; // ms per reading
let _replayAbort = false;

function stopReplay() {
  _replayAbort = true;
}

async function replayReadings(readings) {
  _replayAbort = false;
  const fill    = document.getElementById('replay-fill');
  const label   = document.getElementById('replay-label');
  const speedEl = document.getElementById('replay-speed');
  const stopBtn = document.getElementById('replay-stop');
  speedEl.textContent = `speed: ${_replaySpeed}ms/reading`;
  if (stopBtn) stopBtn.style.display = 'inline-block';

  for (let i = 0; i < readings.length; i++) {
    if (_replayAbort) {
      label.textContent = `⛔ Stopped at reading ${i} / ${readings.length}`;
      fill.style.background = 'var(--yellow)';
      if (stopBtn) stopBtn.style.display = 'none';
      return;
    }

    const r = readings[i];

    // Update sensor panel with raw reading values immediately (visual feedback)
    updateSensors(r);

    // Post to backend (which proxies to ingestion server)
    try {
      await fetch('/api/ingest-one', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(r),
      });
    } catch(e) { console.warn('ingest-one failed', e); }

    // Progress
    const pct = Math.round(((i + 1) / readings.length) * 100);
    fill.style.width  = pct + '%';
    label.textContent = `Reading ${i+1} / ${readings.length}  (${pct}%)  h=${r.hour ?? '?'}:00`;

    await new Promise(res => setTimeout(res, _replaySpeed));
  }

  label.textContent = `✅ Replay complete — ${readings.length} readings sent to AI pipeline`;
  fill.style.background = '#10b981';
  if (stopBtn) stopBtn.style.display = 'none';
}
</script>
</body>
</html>
"""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    source = request.args.get("source", None)   # "live" | "replay" | None=all

    def event_gen():
        try:
            data = build_payload(source)
            yield f"event: init\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
            return

        last_count = len(data.get("timeseries", []))

        while True:
            time.sleep(1)
            try:
                new_data  = build_payload(source)
                new_count = len(new_data.get("timeseries", []))
                if new_count != last_count:
                    last_count = new_count
                    yield f"event: update\ndata: {json.dumps(new_data)}\n\n"
                else:
                    yield ": heartbeat\n\n"
            except GeneratorExit:
                return
            except Exception as e:
                print(f"[SSE] error: {e}")
                yield ": error\n\n"

    return Response(event_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/ingest-one", methods=["POST"])
def ingest_one():
    reading = request.get_json(force=True)
    try:
        r = _req.post(f"{INGEST}/upload",
                      json={"patient": {}, "readings": [reading]},
                      timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/esp32/set", methods=["POST"])
def esp32_set():
    body = request.get_json(force=True) or {}
    try:
        r = _req.post(f"{INGEST}/esp32/set", json=body, timeout=4)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/esp32/silence", methods=["POST"])
def esp32_silence():
    try:
        r = _req.post(f"{INGEST}/esp32/silence", timeout=4)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/esp32/ping")
def esp32_ping():
    try:
        r = _req.get(f"{INGEST}/esp32/ping", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"reachable": False, "reason": str(e)}), 200


@app.route("/api/dashboard")
def dashboard_data():
    return jsonify(build_payload())


if __name__ == "__main__":
    print("[DASHBOARD] AxiAlert — Dementia Monitor Dashboard")
    print("[DASHBOARD] Open: http://localhost:5001")
    print("[DASHBOARD] Ingestion server: http://127.0.0.1:7777")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
