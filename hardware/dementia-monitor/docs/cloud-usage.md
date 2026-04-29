# Cloud Usage (What Is Actually Running)

## Current Runtime

The project currently runs fully without Arduino Cloud:

- ESP32 -> backend ingestion via HTTP POST `/ingest`
- Backend scoring -> ESP32 feedback via HTTP POST `/status`
- Dashboard reads SQLite data from backend

So if your system is working now, it is working through local WiFi + backend services, not through Arduino Cloud.

## Why It Was Confusing

The firmware contains state variables such as:

- `anomalyScore`
- `anomalyLabel`
- `alertStatus`

These are local telemetry state variables. They can be published to Arduino Cloud only if you explicitly enable and initialize cloud code.

## Arduino Cloud Status

`src/thingProperties.h` is now an optional template with placeholder credentials.
No real credentials are stored in the repository anymore.

## If You Want Cloud Later

1. Add valid credentials via secure local config or PlatformIO build flags.
2. Include `thingProperties.h` in firmware.
3. Call `initProperties()` and `ArduinoCloud.begin(...)` in `setup()`.
4. Call `ArduinoCloud.update()` in `loop()`.

Until those steps are done, cloud is intentionally inactive.
