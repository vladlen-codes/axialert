#include <Arduino.h> 
#include <Wire.h>
#include "MAX30105.h"  
#include "heartRate.h"       
#include <MD_MAX72xx.h>       
#include <SPI.h>

#define PIR_PIN        14
#define ACS712_PIN     34 
#define BUZZER_PIN     27
#define MAX7219_DIN    23
#define MAX7219_CLK    18
#define MAX7219_CS      5
#define MAX7219_DEVICES 1
#define ACS712_SENSITIVITY  185.0f   
#define ACS712_VREF         3300.0f  
#define ACS712_ADC_BITS     4095.0f
#define ACS712_ZERO_OFFSET  1650.0f
#define ACS712_SAMPLES      200

MAX30105 particleSensor;

const byte RATE_SIZE = 4;
byte    rates[RATE_SIZE];
byte    rateSpot = 0;
long    lastBeat = 0;
float   beatsPerMinute = 0;
int     beatAvg = 0;

#define SPO2_BUFFER_SIZE 100
uint32_t irBuffer[SPO2_BUFFER_SIZE];
uint32_t redBuffer[SPO2_BUFFER_SIZE];
int32_t  spo2Value  = 0;
int8_t   validSPO2  = 0;
int32_t  heartRate  = 0;
int8_t   validHR    = 0;

MD_MAX72xx mx = MD_MAX72xx(MD_MAX72xx::FC16_HW, MAX7219_CS, MAX7219_DEVICES);

bool     pirState       = false;
bool     prevPirState   = false;
unsigned long pirLastTrigger = 0;
float readCurrentAmps() {
  long   sum    = 0;
  long   sumSq  = 0;

  for (int i = 0; i < ACS712_SAMPLES; i++) {
    int raw = analogRead(ACS712_PIN);
    float mv = (raw / ACS712_ADC_BITS) * ACS712_VREF;
    float delta = mv - ACS712_ZERO_OFFSET;
    sum   += (long)delta;
    sumSq += (long)(delta * delta);
    delayMicroseconds(100);
  }

  float meanSq  = (float)sumSq / ACS712_SAMPLES;
  float rmsAmps = sqrt(meanSq) / ACS712_SENSITIVITY;
  return rmsAmps;
}

bool readVitals(float &hrOut, float &spo2Out) {
  for (int i = 0; i < SPO2_BUFFER_SIZE; i++) {
    while (!particleSensor.available())
      particleSensor.check();

    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }

  if (irBuffer[SPO2_BUFFER_SIZE - 1] < 50000) {
    hrOut   = 0;
    spo2Out = 0;
    return false;
  }

  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, SPO2_BUFFER_SIZE, redBuffer,
    &spo2Value, &validSPO2, &heartRate, &validHR
  );

  hrOut   = validHR   ? (float)heartRate : 0;
  spo2Out = validSPO2 ? (float)spo2Value : 0;
  return true;
}

void scrollText(const char *msg) {
  mx.clear();
  for (uint8_t i = 0; i < strlen(msg) && i < 8; i++) {
    mx.setChar(MAX7219_DEVICES * 8 - 1 - i, msg[i]);
  }
}

void alertBuzzer(int freq, int durationMs) {
  tone(BUZZER_PIN, freq, durationMs);
  delay(durationMs + 50);
  noTone(BUZZER_PIN);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("[BOOT] Dementia Monitor Node — Phase 1 sensor reads");

  pinMode(PIR_PIN, INPUT);
  Serial.println("[PIR] HC-SR501 ready on GPIO " + String(PIR_PIN));

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(ACS712_PIN, INPUT);
  Serial.println("[ACS] ACS712 ready on GPIO " + String(ACS712_PIN));

  Wire.begin();
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[MAX30102] NOT FOUND — check SDA/SCL wiring");
    scrollText("ERR SENS");
    alertBuzzer(500, 1000);
    while (true) delay(1000);
  }

  byte ledBrightness = 60;
  byte sampleAverage = 4;
  byte ledMode = 2;
  int  sampleRate = 100;
  int  pulseWidth = 411;
  int  adcRange = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.enableDIETEMPRDY();
  Serial.println("[MAX30102] Ready");

  mx.begin();
  mx.setIntensity(0, 5);
  mx.clear();
  scrollText("READY");
  Serial.println("[MAX7219] Display ready");

  alertBuzzer(1000, 150);
  alertBuzzer(1200, 150);

  Serial.println("[BOOT] All sensors initialised. Starting read loop.");
  delay(1000);
}

void loop() {
  pirState = digitalRead(PIR_PIN);

  if (pirState && !prevPirState) {
    pirLastTrigger = millis();
    Serial.println("[PIR] Motion DETECTED at " + String(millis()) + " ms");
    scrollText("MOTION");
  }
  if (!pirState && prevPirState) {
    Serial.println("[PIR] Motion CLEARED");
  }
  prevPirState = pirState;

  unsigned long noMotionMs = millis() - pirLastTrigger;
  if (pirLastTrigger > 0 && noMotionMs > 30000 && !pirState) {
    Serial.println("[ALERT] No motion for " + String(noMotionMs / 1000) + "s");
    scrollText("NO MOV");
    alertBuzzer(800, 500);
  }

  float currentA = readCurrentAmps();
  Serial.print("[ACS] Current: ");
  Serial.print(currentA, 3);
  Serial.println(" A");

  float I_on = 0.05f;
  bool applianceOn = (currentA > I_on);
  Serial.println(applianceOn ? "[ACS] Appliance: ON" : "[ACS] Appliance: OFF");

  if (applianceOn) {
    scrollText("APPL ON");
  }

  particleSensor.check();
  if (particleSensor.available()) {
    uint32_t irRaw = particleSensor.getIR();
    particleSensor.nextSample();

    if (irRaw > 50000) {
      Serial.println("[MAX30102] Finger detected, reading vitals...");
      scrollText("VITALS");

      float hr    = 0;
      float spo2  = 0;
      bool  valid = readVitals(hr, spo2);

      if (valid) {
        Serial.print("[MAX30102] HR: ");
        Serial.print(hr, 1);
        Serial.print(" BPM  SpO2: ");
        Serial.print(spo2, 1);
        Serial.println(" %");

        if (hr > 110) {
          Serial.println("[ALERT] Tachycardia flag (HR > 110)");
          alertBuzzer(1200, 300);
          scrollText("TACHY");
        } else if (hr > 0 && hr < 50) {
          Serial.println("[ALERT] Bradycardia flag (HR < 50)");
          alertBuzzer(600, 300);
          scrollText("BRADY");
        }

        if (spo2 > 0 && spo2 < 92) {
          Serial.println("[ALERT] Low SpO2 flag (SpO2 < 92%)");
          alertBuzzer(700, 500);
          scrollText("LO SPO2");
        }
      } else {
        Serial.println("[MAX30102] Finger detected but vitals invalid — hold still");
      }
    }
  }

  Serial.println("-------------------------------");
  delay(2000);
}

