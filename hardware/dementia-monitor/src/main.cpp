/**
 * main.cpp
 * ESP32 — Dementia Monitoring Node
 * Phase 3: ESP32 feedback loop added
 *
 * New in this version:
 *   - AsyncWebServer on port 8080 receives POST /status from backend
 *   - Backend sends { "status": "CRITICAL", "label": "...", "a_global": 0.91 }
 *   - ESP32 updates LED + buzzer based on received AI score
 *   - Local telemetry state anomalyScore updated from backend score
 *
 * New libraries needed in platformio.ini:
 *   ESP Async WebServer + AsyncTCP + ArduinoJson
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <HTTPClient.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <MD_MAX72XX.h>
#include <SPI.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

// Network credentials — defined in secrets.h (NOT committed to git)
#include "secrets.h"
#define STATUS_PORT      8080

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
#define SPO2_MIN_IR       20000
#define HR_HIGH_THRESH    110.0f
#define HR_LOW_THRESH      50.0f
#define SPO2_LOW_THRESH    92.0f

#define NO_MOTION_ALERT_S    3600
#define APPLIANCE_LONG_S     1800
#define DAYTIME_HOUR_START      7
#define DAYTIME_HOUR_END       21

struct FeatureVector {
    unsigned long timestamp_ms;
    int   hour_of_day;
    float hour_sin, hour_cos;
    int   pir_count;
    float pir_active_ratio, time_since_last_motion_s;
    float current_mean, current_std;
    int   appliance_on_flag;
    float on_duration_s;
    int   on_off_switches;
    float hr_mean, hr_std, spo2_mean, spo2_std;
    int   tachycardia_flag, bradycardia_flag, spo2_low_flag, vitals_valid;
    int   motion_and_appliance, motion_no_appliance, no_motion_abnormal_vitals;
};

// Local telemetry state (optionally published to Arduino Cloud)
int    pirCount = 0;        float  pirRatio = 0.0;
float  currentMean = 0.0;   bool   applianceOn = false;
float  heartRate = 0.0;     float  spo2 = 0.0;
float  anomalyScore = 0.0;  String anomalyLabel = "NORMAL";
String alertStatus = "OK";  float  lastMotionSecs = 0.0;
bool   tachycardiaFlag = false; bool spo2LowFlag = false;

// Non-blocking alert state
#define ALERT_DURATION_MS  (40UL * 1000UL)          // 40 seconds
#define ALERT_BLINK_ON_MS  300
#define ALERT_BLINK_OFF_MS 200
bool          _alertActive    = false;
unsigned long _alertEndMs     = 0;
unsigned long _alertToggleMs  = 0;
bool          _alertLedOn     = false;
int           _alertFreq      = 1000;

MAX30105       particleSensor;
MD_MAX72XX     mx = MD_MAX72XX(MD_MAX72XX::FC16_HW, MAX7219_CS, MAX7219_DEVICES);
WiFiUDP        ntpUDP;
NTPClient      timeClient(ntpUDP, "pool.ntp.org", 19800, 60000);
AsyncWebServer statusServer(STATUS_PORT);

// FreeRTOS queue — safe cross-core delivery from async task (Core 0) to loop() (Core 1)
struct AlertMsg { char status[16]; char label[48]; float score; };
QueueHandle_t alertQueue;

int           w_pirCount = 0;
unsigned long w_pirActiveMs = 0, w_pirLastTrigger = 0, w_pirOnStart = 0;
bool          w_pirPrev = false;

#define MAX_CURRENT_SAMPLES 600
struct CurrentSample { float value; bool appOn; };
CurrentSample w_currentSamples[MAX_CURRENT_SAMPLES];
int           w_currentCount = 0;
bool          w_appPrev = false;
int           w_appSwitches = 0;
float         w_appOnMs = 0;
unsigned long w_appOnStart = 0;

#define MAX_VITALS_READINGS 10
float         w_hrReadings[MAX_VITALS_READINGS];
float         w_spo2Readings[MAX_VITALS_READINGS];
int           w_vitalsCount = 0;

uint32_t irBuffer[SPO2_BUFFER_SIZE];
uint32_t redBuffer[SPO2_BUFFER_SIZE];
unsigned long windowStart = 0, lastAcsSample = 0;

#define RETRY_QUEUE_SIZE 5
String retryQueue[RETRY_QUEUE_SIZE];
int    retryCount = 0;

// ─── Helpers ─────────────────────────────────────────────────────────

float readCurrentAmps() {
    float sumSq = 0.0f;  // FIX #17: was `long`, truncated sub-1 values to 0
    for (int i = 0; i < ACS712_SAMPLES; i++) {
        float mv = (analogRead(ACS712_PIN) / ACS712_ADC_BITS) * ACS712_VREF;
        float d  = mv - ACS712_ZERO;
        sumSq   += d * d;  // keep as float — no truncation
        delayMicroseconds(100);
    }
    return sqrt(sumSq / ACS712_SAMPLES) / ACS712_SENS;
}

bool readVitals(float &hrOut, float &spo2Out) {
    for (int i = 0; i < SPO2_BUFFER_SIZE; i++) {
        while (!particleSensor.available()) particleSensor.check();
        redBuffer[i] = particleSensor.getRed();
        irBuffer[i]  = particleSensor.getIR();
        particleSensor.nextSample();
    }
    if (irBuffer[SPO2_BUFFER_SIZE-1] < SPO2_MIN_IR) { hrOut = 0; spo2Out = 0; return false; }
    int32_t sv; int8_t vs; int32_t hv; int8_t vh;
    maxim_heart_rate_and_oxygen_saturation(irBuffer, SPO2_BUFFER_SIZE, redBuffer, &sv, &vs, &hv, &vh);
    hrOut = vh ? (float)hv : 0; spo2Out = vs ? (float)sv : 0;
    bool plausible = (hrOut >= 35.0f && hrOut <= 220.0f && spo2Out >= 70.0f && spo2Out <= 100.0f);
    return (vh && vs && plausible);
}

float arrayMean(float *a, int n) { if(!n) return 0; float s=0; for(int i=0;i<n;i++) s+=a[i]; return s/n; }
float arrayStd(float *a, int n)  {
    if(n<2) return 0; float m=arrayMean(a,n),s=0;
    for(int i=0;i<n;i++) s+=(a[i]-m)*(a[i]-m); return sqrt(s/(n-1));
}
void scrollText(const char *msg) {
    mx.clear();
    if (strlen(msg) > 8) Serial.printf("[SCROLL] Truncating to 8 chars: %.8s\n", msg);
    for(uint8_t i=0;i<strlen(msg)&&i<8;i++) mx.setChar(MAX7219_DEVICES*8-1-i,msg[i]);
}
void alertBuzzer(int f, int ms) { tone(BUZZER_PIN,f,ms); delay(ms+50); noTone(BUZZER_PIN); }

void clearAlertIndicators() {
    _alertActive = false;
    _alertLedOn  = false;
    mx.control(MD_MAX72XX::TEST, MD_MAX72XX::OFF);
    mx.clear();
    noTone(BUZZER_PIN);
}

void startAlert(int freq, unsigned long duration_ms = ALERT_DURATION_MS) {
    _alertActive   = true;
    _alertFreq     = freq;
    _alertEndMs    = millis() + duration_ms;
    _alertToggleMs = 0;
    _alertLedOn    = false;
}

void updateAlert() {
    if (!_alertActive) return;
    unsigned long now = millis();
    if (now >= _alertEndMs) { clearAlertIndicators(); return; }
    if (_alertLedOn && now - _alertToggleMs >= ALERT_BLINK_ON_MS) {
        mx.control(MD_MAX72XX::TEST, MD_MAX72XX::OFF);
        noTone(BUZZER_PIN);
        _alertLedOn    = false;
        _alertToggleMs = now;
    } else if (!_alertLedOn && now - _alertToggleMs >= ALERT_BLINK_OFF_MS) {
        mx.control(MD_MAX72XX::TEST, MD_MAX72XX::ON);
        tone(BUZZER_PIN, _alertFreq);
        _alertLedOn    = true;
        _alertToggleMs = now;
    }
}

// ─── Backend feedback handler (queue-safe across cores) ──────────────

void handleBackendStatus(const String &status, const String &label, float score) {
    Serial.printf("[FEEDBACK] %s | %s | %.3f\n", status.c_str(), label.c_str(), score);
    // FIX #3: Do NOT write global Strings here — this callback runs on Core 0
    // (async task). Writing String objects cross-core without a mutex is UB.
    // loop() (Core 1) receives via alertQueue and updates globals safely.
    AlertMsg msg = {};  // zero-init guarantees null-termination
    strncpy(msg.status, status.c_str(), sizeof(msg.status) - 1);
    strncpy(msg.label,  label.c_str(),  sizeof(msg.label)  - 1);
    msg.score = score;
    xQueueSend(alertQueue, &msg, 0);   // non-blocking; loop() drains it
}

// ─── ESP32 status server ─────────────────────────────────────────────

void setupStatusServer() {
    statusServer.on("/health", HTTP_GET, [](AsyncWebServerRequest *req) {
        req->send(200, "application/json", "{\"status\":\"ok\"}");
    });

    statusServer.on("/status", HTTP_POST,
        [](AsyncWebServerRequest *req) {},
        NULL,
        [](AsyncWebServerRequest *req, uint8_t *data, size_t len, size_t, size_t) {
            JsonDocument doc;
            if (deserializeJson(doc, data, len)) {
                req->send(400, "application/json", "{\"error\":\"bad json\"}");
                return;
            }
            handleBackendStatus(
                doc["status"]   | "OK",
                doc["label"]    | "NORMAL",
                doc["a_global"] | 0.0f
            );
            req->send(200, "application/json", "{\"received\":true}");
        }
    );

    statusServer.begin();
    Serial.printf("[STATUS] Listening on http://%s:%d\n",
                  WiFi.localIP().toString().c_str(), STATUS_PORT);
    Serial.printf("[STATUS] Backend should POST to: http://%s:%d/status\n",
                  WiFi.localIP().toString().c_str(), STATUS_PORT);
}

// ─── Telemetry + HTTP helpers ────────────────────────────────────────

void syncTelemetryState(const FeatureVector &fv) {
    pirCount = fv.pir_count; pirRatio = fv.pir_active_ratio;
    currentMean = fv.current_mean; applianceOn = (fv.appliance_on_flag==1);
    heartRate = fv.hr_mean; spo2 = fv.spo2_mean;
    lastMotionSecs = fv.time_since_last_motion_s;
    tachycardiaFlag = (fv.tachycardia_flag==1); spo2LowFlag = (fv.spo2_low_flag==1);
}

void setLocalAlertState(const String &type, float value) {
    alertStatus = "CRITICAL"; anomalyLabel = type; anomalyScore = 1.0;
    Serial.printf("[CLOUD] Alert: %s (%.1f)\n", type.c_str(), value);
}

bool postJSON(const String &json) {
    if (WiFi.status() != WL_CONNECTED) return false;
    HTTPClient http;
    http.begin(BACKEND_URL);
    http.addHeader("Content-Type","application/json");
    http.setTimeout(5000);
    int code = http.POST(json);
    bool ok = (code==200||code==201);
    if (ok) Serial.println("[HTTP] OK — " + http.getString());
    else    Serial.printf("[HTTP] Failed: %d\n", code);
    http.end(); return ok;
}

void flushRetryQueue() {
    if (!retryCount || WiFi.status()!=WL_CONNECTED) return;
    int sent=0;
    for(int i=0;i<retryCount;i++) { if(postJSON(retryQueue[i])) sent++; else break; }
    for(int i=0;i<retryCount-sent;i++) retryQueue[i]=retryQueue[i+sent];
    retryCount-=sent;
    if(sent) Serial.printf("[RETRY] Flushed %d\n",sent);
}

void resetWindow() {
    w_pirCount=0; w_pirActiveMs=0; w_appSwitches=0; w_appOnMs=0;
    w_currentCount=0; w_vitalsCount=0; w_appPrev=false; windowStart=millis();
}

// ─── Compute + transmit ──────────────────────────────────────────────

void computeAndTransmit() {
    FeatureVector fv;
    unsigned long now = millis(), dur = now - windowStart;

    fv.timestamp_ms = now;
    fv.hour_of_day  = timeClient.getHours();
    fv.hour_sin     = sin(2.0f*M_PI*fv.hour_of_day/24.0f);
    fv.hour_cos     = cos(2.0f*M_PI*fv.hour_of_day/24.0f);

    if (w_pirPrev && w_pirOnStart) w_pirActiveMs += (now-w_pirOnStart);
    fv.pir_count        = w_pirCount;
    fv.pir_active_ratio = dur ? (float)w_pirActiveMs/dur : 0;
    fv.time_since_last_motion_s = w_pirLastTrigger ? (now-w_pirLastTrigger)/1000.0f : -1;

    if (w_appPrev && w_appOnStart) w_appOnMs += (now-w_appOnStart);
    fv.on_duration_s=w_appOnMs/1000.0f; fv.on_off_switches=w_appSwitches;

    if (w_currentCount > 0) {
        float v[MAX_CURRENT_SAMPLES]; bool anyOn=false;
        for(int i=0;i<w_currentCount;i++){v[i]=w_currentSamples[i].value; if(w_currentSamples[i].appOn) anyOn=true;}
        fv.current_mean=arrayMean(v,w_currentCount); fv.current_std=arrayStd(v,w_currentCount);
        fv.appliance_on_flag=anyOn?1:0;
    } else { fv.current_mean=fv.current_std=0; fv.appliance_on_flag=0; }

    if (w_vitalsCount > 0) {
        fv.hr_mean=arrayMean(w_hrReadings,w_vitalsCount); fv.hr_std=arrayStd(w_hrReadings,w_vitalsCount);
        fv.spo2_mean=arrayMean(w_spo2Readings,w_vitalsCount); fv.spo2_std=arrayStd(w_spo2Readings,w_vitalsCount);
        fv.tachycardia_flag=(fv.hr_mean>HR_HIGH_THRESH)?1:0;
        fv.bradycardia_flag=(fv.hr_mean<HR_LOW_THRESH&&fv.hr_mean>0)?1:0;
        fv.spo2_low_flag=(fv.spo2_mean<SPO2_LOW_THRESH&&fv.spo2_mean>0)?1:0;
        fv.vitals_valid=1;
    } else { fv.hr_mean=fv.hr_std=fv.spo2_mean=fv.spo2_std=0; fv.tachycardia_flag=fv.bradycardia_flag=fv.spo2_low_flag=fv.vitals_valid=0; }

    bool pa=(fv.pir_active_ratio>0.05f), ao=(fv.appliance_on_flag==1);
    bool av=(fv.tachycardia_flag||fv.bradycardia_flag||fv.spo2_low_flag);
    fv.motion_and_appliance=(pa&&ao)?1:0; fv.motion_no_appliance=(pa&&!ao)?1:0;
    fv.no_motion_abnormal_vitals=(!pa&&av&&fv.vitals_valid)?1:0;

    bool day=(fv.hour_of_day>=DAYTIME_HOUR_START&&fv.hour_of_day<=DAYTIME_HOUR_END);
    if(fv.time_since_last_motion_s>NO_MOTION_ALERT_S&&day){ setLocalAlertState("CRITICAL:PROLONGED_INACTIVITY",fv.time_since_last_motion_s); startAlert(1000); }
    if(fv.on_duration_s>APPLIANCE_LONG_S){ setLocalAlertState("WARNING:APPLIANCE_LONG",fv.on_duration_s); startAlert(900); }
    if(fv.tachycardia_flag){ setLocalAlertState("WARNING:TACHYCARDIA",fv.hr_mean); startAlert(1200); }
    if(fv.bradycardia_flag){ setLocalAlertState("WARNING:BRADYCARDIA",fv.hr_mean); startAlert(700); }
    if(fv.spo2_low_flag)   { setLocalAlertState("WARNING:LOW_SPO2",fv.spo2_mean);  startAlert(800); }

    String j="{";
    j+="\"ts\":"+String(fv.timestamp_ms)+",\"hour\":"+String(fv.hour_of_day)+",";
    j+="\"hour_sin\":"+String(fv.hour_sin,4)+",\"hour_cos\":"+String(fv.hour_cos,4)+",";
    j+="\"pir_count\":"+String(fv.pir_count)+",\"pir_ratio\":"+String(fv.pir_active_ratio,4)+",";
    j+="\"no_motion_s\":"+String(fv.time_since_last_motion_s,1)+",";
    j+="\"cur_mean\":"+String(fv.current_mean,4)+",\"cur_std\":"+String(fv.current_std,4)+",";
    j+="\"app_on\":"+String(fv.appliance_on_flag)+",\"app_on_s\":"+String(fv.on_duration_s,1)+",";
    j+="\"app_switches\":"+String(fv.on_off_switches)+",";
    j+="\"hr_mean\":"+String(fv.hr_mean,1)+",\"hr_std\":"+String(fv.hr_std,2)+",";
    j+="\"spo2_mean\":"+String(fv.spo2_mean,1)+",\"spo2_std\":"+String(fv.spo2_std,2)+",";
    j+="\"tachy\":"+String(fv.tachycardia_flag)+",\"brady\":"+String(fv.bradycardia_flag)+",";
    j+="\"spo2_low\":"+String(fv.spo2_low_flag)+",\"vitals_valid\":"+String(fv.vitals_valid)+",";
    j+="\"mot_and_app\":"+String(fv.motion_and_appliance)+",\"mot_no_app\":"+String(fv.motion_no_appliance)+",";
    j+="\"nomot_abnvit\":"+String(fv.no_motion_abnormal_vitals)+"}";

    Serial.println("\n[FV] "+j);
    syncTelemetryState(fv);
    flushRetryQueue();
    if(postJSON(j)){ /* backend will send feedback — don't clear here */ }
    else {
        if(retryCount<RETRY_QUEUE_SIZE) {
            retryQueue[retryCount++]=j;
        } else {
            Serial.println("[RETRY] Queue full — dropping oldest entry (data loss)");
            for(int i=0;i<RETRY_QUEUE_SIZE-1;i++) retryQueue[i]=retryQueue[i+1];
            retryQueue[RETRY_QUEUE_SIZE-1]=j;
        }
        startAlert(900, 5000);  // 5s blink on send failure, not 3 min
    }
}

// ─── Setup / Loop ────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200); delay(500);
    Serial.println("[BOOT] Phase 3 — feedback loop enabled");

    alertQueue = xQueueCreate(10, sizeof(AlertMsg));
    configASSERT(alertQueue != NULL);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.printf("[WiFi] Connecting to \"%s\"", WIFI_SSID);
    for (int i = 0; WiFi.status() != WL_CONNECTED && i < 40; i++) {
        delay(500); Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] Connected — IP: %s  RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        Serial.printf("[WiFi] FAILED (status=%d). Check SSID/password and that network is 2.4GHz.\n",
                      WiFi.status());
        Serial.println("[WiFi] Status codes: 1=NO_SSID, 3=CONNECTED, 4=CONNECT_FAILED, 6=DISCONNECTED");
    }

    pinMode(PIR_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    analogReadResolution(12); analogSetAttenuation(ADC_11db);
    pinMode(ACS712_PIN, INPUT);

    Wire.begin();
    if(!particleSensor.begin(Wire, I2C_SPEED_FAST)){ Serial.println("[ERROR] MAX30102"); while(1) delay(1000); }
    particleSensor.setup(60,4,2,100,411,4096);

    mx.begin(); mx.control(MD_MAX72XX::INTENSITY, 5); clearAlertIndicators();

    resetWindow(); lastAcsSample=millis();
    Serial.printf("[BOOT] Window: %lu ms\n", WINDOW_MS);
}

void loop() {
    unsigned long now = millis();

    updateAlert();

    AlertMsg msg;
    if (xQueueReceive(alertQueue, &msg, 0) == pdTRUE) {
        String st(msg.status), lb(msg.label);
        // FIX #3: update globals here on Core 1 — safe, no cross-core race
        anomalyScore = msg.score;
        anomalyLabel = lb;
        alertStatus  = st;
        if (st == "CRITICAL") {
            startAlert(1000);
        } else if (st == "WARNING") {
            startAlert(800);
        } else if (st == "SILENCED") {
            clearAlertIndicators();  // user pressed silence button — stop immediately
        }
        // "OK" from regular scoring does nothing — active alert runs its full duration
    }

    static bool serverStarted = false;
    if (!serverStarted && WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] IP: %s\n", WiFi.localIP().toString().c_str());
        timeClient.begin(); timeClient.update();
        setupStatusServer();
        serverStarted = true;
    }

    bool pirNow=digitalRead(PIR_PIN);
    if( pirNow&&!w_pirPrev){ w_pirCount++; w_pirLastTrigger=now; w_pirOnStart=now; }
    if(!pirNow&& w_pirPrev){ w_pirActiveMs+=(now-w_pirOnStart); }
    w_pirPrev=pirNow;

    if(now-lastAcsSample>=ACS_SAMPLE_MS&&w_currentCount<MAX_CURRENT_SAMPLES){
        float a=readCurrentAmps(); bool on=(a>I_ON_THRESHOLD);
        w_currentSamples[w_currentCount++]={a,on};
        if( on&&!w_appPrev){ w_appSwitches++; w_appOnStart=now; }
        if(!on&& w_appPrev){ w_appSwitches++; w_appOnMs+=(now-w_appOnStart); }
        w_appPrev=on; lastAcsSample=now;
    }

    particleSensor.check();
    if(particleSensor.available()){
        uint32_t ir=particleSensor.getIR(); particleSensor.nextSample();
        if(ir>SPO2_MIN_IR&&w_vitalsCount<MAX_VITALS_READINGS){
            float hr,sp;
            if(readVitals(hr,sp)&&hr>0&&sp>0){
                w_hrReadings[w_vitalsCount]=hr; w_spo2Readings[w_vitalsCount]=sp; w_vitalsCount++;
                Serial.printf("[VITALS] HR:%.1f SpO2:%.1f\n",hr,sp);
            }
        }
    }

    timeClient.update();

    if(now-windowStart>=WINDOW_MS){ computeAndTransmit(); resetWindow(); clearAlertIndicators(); }
}
