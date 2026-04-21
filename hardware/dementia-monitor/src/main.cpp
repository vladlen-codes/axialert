#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <MD_MAX72xx.h>
#include <SPI.h>

#define WIFI_SSID       "Vladlen's S24"
#define WIFI_PASS       "9021331219"
#define BACKEND_URL     "http://192.168.1.100:5000/ingest"

#define PIR_PIN          14
#define ACS712_PIN       34
#define BUZZER_PIN       27
#define MAX7219_DIN      23
#define MAX7219_CLK      18
#define MAX7219_CS        5
#define MAX7219_DEVICES   1

#define WINDOW_MS        (5UL * 60UL * 1000UL)
#define ACS_SAMPLE_MS    500
#define ACS712_SAMPLES   200
#define ACS712_SENS      185.0f
#define ACS712_VREF      3300.0f
#define ACS712_ADC_BITS  4095.0f
#define ACS712_ZERO      1650.0f
#define I_ON_THRESHOLD   0.05f

#define SPO2_BUFFER_SIZE  100
#define HR_HIGH_THRESH    110.0f
#define HR_LOW_THRESH      50.0f
#define SPO2_LOW_THRESH    92.0f

#define NO_MOTION_ALERT_S     3600
#define APPLIANCE_LONG_S      1800
#define DAYTIME_HOUR_START       7
#define DAYTIME_HOUR_END        21

struct FeatureVector {
  unsigned long timestamp_ms;
  int           hour_of_day;
  float         hour_sin;
  float         hour_cos;
  int           pir_count;
  float         pir_active_ratio;
  float         time_since_last_motion_s;
  float         current_mean;
  float         current_std;
  int           appliance_on_flag;
  float         on_duration_s;
  int           on_off_switches;
  float         hr_mean;
  float         hr_std;
  float         spo2_mean;
  float         spo2_std;
  int           tachycardia_flag;
  int           bradycardia_flag;
  int           spo2_low_flag;
  int           vitals_valid;
  int           motion_and_appliance;
  int           motion_no_appliance;
  int           no_motion_abnormal_vitals;
};

MAX30105    particleSensor;
MD_MAX72xx  mx = MD_MAX72xx(MD_MAX72xx::FC16_HW, MAX7219_CS, MAX7219_DEVICES);
WiFiUDP     ntpUDP;
NTPClient   timeClient(ntpUDP, "pool.ntp.org", 19800, 60000);

int           w_pirCount      = 0;
unsigned long w_pirActiveMs   = 0;
unsigned long w_pirLastTrigger= 0;
bool          w_pirPrev       = false;
unsigned long w_pirOnStart    = 0;

#define MAX_CURRENT_SAMPLES 600
struct CurrentSample { float value; bool appOn; };
CurrentSample w_currentSamples[MAX_CURRENT_SAMPLES];
int           w_currentCount  = 0;
bool          w_appPrev       = false;
int           w_appSwitches   = 0;
float         w_appOnMs       = 0;
unsigned long w_appOnStart    = 0;

#define MAX_VITALS_READINGS 10
float         w_hrReadings[MAX_VITALS_READINGS];
float         w_spo2Readings[MAX_VITALS_READINGS];
int           w_vitalsCount   = 0;

uint32_t irBuffer[SPO2_BUFFER_SIZE];
uint32_t redBuffer[SPO2_BUFFER_SIZE];

unsigned long windowStart    = 0;
unsigned long lastAcsSample  = 0;

#define RETRY_QUEUE_SIZE 5
String retryQueue[RETRY_QUEUE_SIZE];
int    retryCount = 0;

float readCurrentAmps() {
  long sumSq = 0;
  for (int i = 0; i < ACS712_SAMPLES; i++) {
    int   raw   = analogRead(ACS712_PIN);
    float mv    = (raw / ACS712_ADC_BITS) * ACS712_VREF;
    float delta = mv - ACS712_ZERO;
    sumSq += (long)(delta * delta);
    delayMicroseconds(100);
  }
  return sqrt((float)sumSq / ACS712_SAMPLES) / ACS712_SENS;
}

bool readVitals(float &hrOut, float &spo2Out) {
  for (int i = 0; i < SPO2_BUFFER_SIZE; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }
  if (irBuffer[SPO2_BUFFER_SIZE - 1] < 50000) { hrOut = 0; spo2Out = 0; return false; }
  int32_t spo2Val; int8_t validSpo2;
  int32_t hrVal;   int8_t validHr;
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, SPO2_BUFFER_SIZE, redBuffer,
    &spo2Val, &validSpo2, &hrVal, &validHr
  );
  hrOut   = validHr   ? (float)hrVal   : 0;
  spo2Out = validSpo2 ? (float)spo2Val : 0;
  return (validHr && validSpo2);
}

float arrayMean(float *arr, int n) {
  if (n == 0) return 0;
  float s = 0;
  for (int i = 0; i < n; i++) s += arr[i];
  return s / n;
}

float arrayStd(float *arr, int n) {
  if (n < 2) return 0;
  float mean = arrayMean(arr, n);
  float sq = 0;
  for (int i = 0; i < n; i++) sq += (arr[i] - mean) * (arr[i] - mean);
  return sqrt(sq / (n - 1));
}

void scrollText(const char *msg) {
  mx.clear();
  for (uint8_t i = 0; i < strlen(msg) && i < 8; i++)
    mx.setChar(MAX7219_DEVICES * 8 - 1 - i, msg[i]);
}

void alertBuzzer(int freq, int durationMs) {
  tone(BUZZER_PIN, freq, durationMs);
  delay(durationMs + 50);
  noTone(BUZZER_PIN);
}


void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  scrollText("WIFI...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected. IP: %s\n", WiFi.localIP().toString().c_str());
    scrollText("WIFI OK");
    alertBuzzer(1200, 150);
    timeClient.begin();
    timeClient.update();
  } else {
    Serial.println("\n[WiFi] FAILED — running offline. Will retry each window.");
    scrollText("NO WIFI");
  }
}


bool postJSON(const String &json) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] No Wi-Fi — queuing for retry");
    return false;
  }
  HTTPClient http;
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  int code = http.POST(json);
  if (code == 200 || code == 201) {
    Serial.printf("[HTTP] POST OK (%d)\n", code);
    String response = http.getString();
    Serial.println("[HTTP] Response: " + response);
    http.end();
    return true;
  } else {
    Serial.printf("[HTTP] POST FAILED — code %d\n", code);
    http.end();
    return false;
  }
}

void flushRetryQueue() {
  if (retryCount == 0 || WiFi.status() != WL_CONNECTED) return;
  Serial.printf("[RETRY] Flushing %d queued vectors\n", retryCount);
  int sent = 0;
  for (int i = 0; i < retryCount; i++) {
    if (postJSON(retryQueue[i])) {
      sent++;
    } else {
      break;
    }
  }
  for (int i = 0; i < retryCount - sent; i++)
    retryQueue[i] = retryQueue[i + sent];
  retryCount -= sent;
}


void resetWindow() {
  w_pirCount     = 0;
  w_pirActiveMs  = 0;
  w_appSwitches  = 0;
  w_appOnMs      = 0;
  w_currentCount = 0;
  w_vitalsCount  = 0;
  w_appPrev      = false;
  windowStart    = millis();
}


void computeAndTransmit() {
  FeatureVector fv;
  unsigned long now = millis();
  unsigned long windowDurationMs = now - windowStart;

  fv.timestamp_ms = now;
  if (WiFi.status() == WL_CONNECTED) {
    timeClient.update();
    fv.hour_of_day = timeClient.getHours();
  } else {
    fv.hour_of_day = (int)((now / 3600000UL) % 24);
  }
  fv.hour_sin = sin(2.0f * M_PI * fv.hour_of_day / 24.0f);
  fv.hour_cos = cos(2.0f * M_PI * fv.hour_of_day / 24.0f);

  if (w_pirPrev && w_pirOnStart > 0)
    w_pirActiveMs += (now - w_pirOnStart);
  fv.pir_count        = w_pirCount;
  fv.pir_active_ratio = windowDurationMs > 0
                        ? (float)w_pirActiveMs / windowDurationMs : 0;
  fv.time_since_last_motion_s = w_pirLastTrigger > 0
                                ? (now - w_pirLastTrigger) / 1000.0f : -1;

  if (w_appPrev && w_appOnStart > 0)
    w_appOnMs += (now - w_appOnStart);
  fv.on_duration_s   = w_appOnMs / 1000.0f;
  fv.on_off_switches = w_appSwitches;
  if (w_currentCount > 0) {
    float vals[MAX_CURRENT_SAMPLES];
    bool  anyOn = false;
    for (int i = 0; i < w_currentCount; i++) {
      vals[i] = w_currentSamples[i].value;
      if (w_currentSamples[i].appOn) anyOn = true;
    }
    fv.current_mean      = arrayMean(vals, w_currentCount);
    fv.current_std       = arrayStd(vals,  w_currentCount);
    fv.appliance_on_flag = anyOn ? 1 : 0;
  } else {
    fv.current_mean = fv.current_std = 0;
    fv.appliance_on_flag = 0;
  }

  if (w_vitalsCount > 0) {
    fv.hr_mean   = arrayMean(w_hrReadings,   w_vitalsCount);
    fv.hr_std    = arrayStd(w_hrReadings,    w_vitalsCount);
    fv.spo2_mean = arrayMean(w_spo2Readings, w_vitalsCount);
    fv.spo2_std  = arrayStd(w_spo2Readings,  w_vitalsCount);
    fv.tachycardia_flag = (fv.hr_mean   > HR_HIGH_THRESH)            ? 1 : 0;
    fv.bradycardia_flag = (fv.hr_mean   < HR_LOW_THRESH && fv.hr_mean > 0) ? 1 : 0;
    fv.spo2_low_flag    = (fv.spo2_mean < SPO2_LOW_THRESH && fv.spo2_mean > 0) ? 1 : 0;
    fv.vitals_valid     = 1;
  } else {
    fv.hr_mean = fv.hr_std = fv.spo2_mean = fv.spo2_std = 0;
    fv.tachycardia_flag = fv.bradycardia_flag = fv.spo2_low_flag = 0;
    fv.vitals_valid = 0;
  }

  bool pirActive      = (fv.pir_active_ratio > 0.05f);
  bool appOn          = (fv.appliance_on_flag == 1);
  bool abnormalVitals = (fv.tachycardia_flag || fv.bradycardia_flag || fv.spo2_low_flag);
  fv.motion_and_appliance      = (pirActive && appOn)                             ? 1 : 0;
  fv.motion_no_appliance       = (pirActive && !appOn)                            ? 1 : 0;
  fv.no_motion_abnormal_vitals = (!pirActive && abnormalVitals && fv.vitals_valid)? 1 : 0;


  bool isDaytime = (fv.hour_of_day >= DAYTIME_HOUR_START &&
                    fv.hour_of_day <= DAYTIME_HOUR_END);

  if (fv.time_since_last_motion_s > NO_MOTION_ALERT_S && isDaytime) {
    Serial.println("[ALERT] No motion > 1hr during daytime");
    scrollText("NO MOV");
    alertBuzzer(800, 600);
    delay(300);
    alertBuzzer(800, 600);
  }
  if (fv.on_duration_s > APPLIANCE_LONG_S) {
    Serial.println("[ALERT] Appliance ON > 30 min");
    scrollText("APL LNG");
    alertBuzzer(900, 600);
  }
  if (fv.tachycardia_flag) {
    Serial.println("[ALERT] Tachycardia");
    scrollText("TACHY");
    alertBuzzer(1200, 400);
  }
  if (fv.bradycardia_flag) {
    Serial.println("[ALERT] Bradycardia");
    scrollText("BRADY");
    alertBuzzer(600, 400);
  }
  if (fv.spo2_low_flag) {
    Serial.println("[ALERT] Low SpO2");
    scrollText("LO SP02");
    alertBuzzer(700, 600);
  }

  String json = "{";
  json += "\"ts\":"           + String(fv.timestamp_ms)              + ",";
  json += "\"hour\":"         + String(fv.hour_of_day)               + ",";
  json += "\"hour_sin\":"     + String(fv.hour_sin,    4)            + ",";
  json += "\"hour_cos\":"     + String(fv.hour_cos,    4)            + ",";
  json += "\"pir_count\":"    + String(fv.pir_count)                 + ",";
  json += "\"pir_ratio\":"    + String(fv.pir_active_ratio, 4)       + ",";
  json += "\"no_motion_s\":"  + String(fv.time_since_last_motion_s, 1) + ",";
  json += "\"cur_mean\":"     + String(fv.current_mean,  4)          + ",";
  json += "\"cur_std\":"      + String(fv.current_std,   4)          + ",";
  json += "\"app_on\":"       + String(fv.appliance_on_flag)         + ",";
  json += "\"app_on_s\":"     + String(fv.on_duration_s, 1)          + ",";
  json += "\"app_switches\":" + String(fv.on_off_switches)           + ",";
  json += "\"hr_mean\":"      + String(fv.hr_mean,    1)             + ",";
  json += "\"hr_std\":"       + String(fv.hr_std,     2)             + ",";
  json += "\"spo2_mean\":"    + String(fv.spo2_mean,  1)             + ",";
  json += "\"spo2_std\":"     + String(fv.spo2_std,   2)             + ",";
  json += "\"tachy\":"        + String(fv.tachycardia_flag)          + ",";
  json += "\"brady\":"        + String(fv.bradycardia_flag)          + ",";
  json += "\"spo2_low\":"     + String(fv.spo2_low_flag)             + ",";
  json += "\"vitals_valid\":" + String(fv.vitals_valid)              + ",";
  json += "\"mot_and_app\":"  + String(fv.motion_and_appliance)      + ",";
  json += "\"mot_no_app\":"   + String(fv.motion_no_appliance)       + ",";
  json += "\"nomot_abnvit\":" + String(fv.no_motion_abnormal_vitals) ;
  json += "}";

  Serial.println("\n[FEATURE VECTOR] " + json);

  scrollText("SEND...");

  flushRetryQueue();

  bool success = postJSON(json);
  if (success) {
    scrollText("SENT OK");
    alertBuzzer(1000, 100);
  } else {
    if (retryCount < RETRY_QUEUE_SIZE) {
      retryQueue[retryCount++] = json;
      Serial.printf("[RETRY] Queued. Queue size: %d\n", retryCount);
    } else {
      Serial.println("[RETRY] Queue full — dropping oldest vector");
      for (int i = 0; i < RETRY_QUEUE_SIZE - 1; i++)
        retryQueue[i] = retryQueue[i + 1];
      retryQueue[RETRY_QUEUE_SIZE - 1] = json;
    }
    scrollText("NO SEND");
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("[BOOT] Phase 1 Step 3 — Transmit + Rules");

  pinMode(PIR_PIN, INPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(ACS712_PIN, INPUT);

  Wire.begin();
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERROR] MAX30102 not found — check wiring");
    while (true) delay(1000);
  }
  particleSensor.setup(60, 4, 2, 100, 411, 4096);

  mx.begin();
  mx.setIntensity(0, 5);
  mx.clear();
  scrollText("BOOT...");

  connectWiFi();

  tone(BUZZER_PIN, 1000, 150);
  delay(200);
  noTone(BUZZER_PIN);

  resetWindow();
  lastAcsSample = millis();
  Serial.printf("[BOOT] Window: %lu ms. Ready.\n", WINDOW_MS);
  scrollText("READY");
}

void loop() {
  unsigned long now = millis();

  bool pirNow = digitalRead(PIR_PIN);
  if (pirNow && !w_pirPrev) {
    w_pirCount++;
    w_pirLastTrigger = now;
    w_pirOnStart     = now;
  }
  if (!pirNow && w_pirPrev)
    w_pirActiveMs += (now - w_pirOnStart);
  w_pirPrev = pirNow;

  if (now - lastAcsSample >= ACS_SAMPLE_MS && w_currentCount < MAX_CURRENT_SAMPLES) {
    float amps = readCurrentAmps();
    bool  appOn = (amps > I_ON_THRESHOLD);
    w_currentSamples[w_currentCount++] = { amps, appOn };
    if (appOn  && !w_appPrev) { w_appSwitches++; w_appOnStart = now; }
    if (!appOn &&  w_appPrev) { w_appSwitches++; w_appOnMs += (now - w_appOnStart); }
    w_appPrev     = appOn;
    lastAcsSample = now;
  }

  particleSensor.check();
  if (particleSensor.available()) {
    uint32_t ir = particleSensor.getIR();
    particleSensor.nextSample();
    if (ir > 50000 && w_vitalsCount < MAX_VITALS_READINGS) {
      float hr, spo2;
      if (readVitals(hr, spo2) && hr > 0 && spo2 > 0) {
        w_hrReadings[w_vitalsCount]   = hr;
        w_spo2Readings[w_vitalsCount] = spo2;
        w_vitalsCount++;
        Serial.printf("[VITALS] HR: %.1f  SpO2: %.1f\n", hr, spo2);
      }
    }
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected — reconnecting...");
    connectWiFi();
  }

  if (now - windowStart >= WINDOW_MS) {
    scrollText("COMPUTE");
    computeAndTransmit();
    resetWindow();
    scrollText("READY");
  }
}