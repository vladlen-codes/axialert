/**
 * thingProperties.h
 * Arduino Cloud — OPTIONAL configuration for Thing: DementiaMonitor
 *
 * IMPORTANT:
 *   This file is optional and is NOT used unless your sketch explicitly
 *   includes it and calls ArduinoCloud.begin()/ArduinoCloud.update().
 *
 * SETUP (only if you want Arduino Cloud publishing):
 *   1. Go to cloud.arduino.cc
 *   2. Open your DementiaMonitor Thing
 *   3. Go to "Setup" tab -> click your device
 *   4. Provide Device ID and Secret via build flags or defines
 *   5. Add these variables in the Arduino Cloud dashboard
 *      (Thing → Variables tab → Add Variable):
 *
 *   Variable Name        Type      Permission    Update
 *   ─────────────────────────────────────────────────────
 *   pirCount             int       Read Only     On Change
 *   pirRatio             float     Read Only     On Change
 *   currentMean          float     Read Only     On Change
 *   applianceOn          bool      Read Only     On Change
 *   heartRate            float     Read Only     On Change
 *   spo2                 float     Read Only     On Change
 *   anomalyScore         float     Read Only     On Change
 *   anomalyLabel         String    Read Only     On Change
 *   alertStatus          String    Read Only     On Change
 *   lastMotionSecs       float     Read Only     On Change
 *   tachycardiaFlag      bool      Read Only     On Change
 *   spo2LowFlag          bool      Read Only     On Change
 *
 * After adding all variables, click "Sketch" tab -> Arduino Cloud
 * will show you the auto-generated thingProperties snippet.
 * The one below is pre-written to match and uses placeholders by default.
 */

#pragma once
#include <ArduinoIoTCloud.h>
#include <Arduino_ConnectionHandler.h>

// Define these through PlatformIO build_flags or local private headers.
#ifndef DEVICE_ID
#define DEVICE_ID "REPLACE_WITH_DEVICE_ID"
#endif

#ifndef DEVICE_SECRET
#define DEVICE_SECRET "REPLACE_WITH_DEVICE_SECRET"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "REPLACE_WITH_WIFI_SSID"
#endif

#ifndef WIFI_PASS
#define WIFI_PASS "REPLACE_WITH_WIFI_PASS"
#endif

// ─── Cloud variables — synced to DementiaMonitor Thing ───────────────
// Declared here, defined in main.cpp
extern int    pirCount;
extern float  pirRatio;
extern float  currentMean;
extern bool   applianceOn;
extern float  heartRate;
extern float  spo2;
extern float  anomalyScore;
extern String anomalyLabel;
extern String alertStatus;
extern float  lastMotionSecs;
extern bool   tachycardiaFlag;
extern bool   spo2LowFlag;

// ─── Connection handler ──────────────────────────────────────────────
WiFiConnectionHandler ArduinoIoTPreferredConnection(WIFI_SSID, WIFI_PASS);

// ─── Register all cloud variables ───────────────────────────────────
inline void initProperties() {
    ArduinoCloud.setBoardId(DEVICE_ID);
    ArduinoCloud.setSecretDeviceKey(DEVICE_SECRET);

    ArduinoCloud.addProperty(pirCount,        Permission::Read).publishOnChange(0);
    ArduinoCloud.addProperty(pirRatio,        Permission::Read).publishOnChange(0.01f);
    ArduinoCloud.addProperty(currentMean,     Permission::Read).publishOnChange(0.001f);
    ArduinoCloud.addProperty(applianceOn,     Permission::Read).publishOnChange(0);
    ArduinoCloud.addProperty(heartRate,       Permission::Read).publishOnChange(0.5f);
    ArduinoCloud.addProperty(spo2,            Permission::Read).publishOnChange(0.5f);
    ArduinoCloud.addProperty(anomalyScore,    Permission::Read).publishOnChange(0.01f);
    ArduinoCloud.addProperty(anomalyLabel,    Permission::Read).publishOnChange(0);
    ArduinoCloud.addProperty(alertStatus,     Permission::Read).publishOnChange(0);
    ArduinoCloud.addProperty(lastMotionSecs,  Permission::Read).publishOnChange(1.0f);
    ArduinoCloud.addProperty(tachycardiaFlag, Permission::Read).publishOnChange(0);
    ArduinoCloud.addProperty(spo2LowFlag,     Permission::Read).publishOnChange(0);
}