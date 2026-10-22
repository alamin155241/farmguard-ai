#include "secrets.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// =====================================================
// WIFI
// =====================================================
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// =====================================================
// MQTT
// =====================================================
const char* mqtt_server = "broker.emqx.io";
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

// =====================================================
// SENSOR PINS
// =====================================================
#define PIR_PIN       27
#define DHT_PIN        4
#define MQ_PIN        34
#define LDR_PIN       35
#define BUZZER_PIN    26

#define DHT_TYPE DHT11
DHT dht(DHT_PIN, DHT_TYPE);

// =====================================================
// SETTINGS
// =====================================================

// Calibrated from your test:
// Laser clear around 48–52
// Beam blocked around 16–25
#define LDR_THRESHOLD 30

#define FUSION_WINDOW     3000
#define ALERT_HOLD_TIME   3000
#define MQTT_INTERVAL      500
#define DHT_INTERVAL      2000

// =====================================================
// VARIABLES
// =====================================================
bool previousMotion = false;
bool previousBeamBlocked = false;

unsigned long lastPirEvent = 0;
unsigned long lastBeamEvent = 0;
unsigned long confirmedUntil = 0;

unsigned long lastMqttTime = 0;
unsigned long lastDhtTime = 0;

float temperature = 0;

String previousState = "";

char mqttTopic[80];
char clientId[50];

// =====================================================
// CONNECT WIFI
// =====================================================
void connectWiFi() {

  Serial.print("Connecting to WiFi");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi CONNECTED");

  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

// =====================================================
// CONNECT MQTT
// =====================================================
void connectMQTT() {

  while (!client.connected()) {

    Serial.print("Connecting to MQTT...");

    if (client.connect(clientId)) {

      Serial.println("CONNECTED");

      client.publish(
        mqttTopic,
        "{\"status\":\"FarmGuard-AI ONLINE\"}"
      );

      Serial.println("FarmGuard online message published!");
    }

    else {

      Serial.print("FAILED, state=");
      Serial.println(client.state());

      delay(2000);
    }
  }
}

// =====================================================
// PUBLISH SENSOR DATA
// =====================================================
void publishData(
  bool motion,
  int ldrValue,
  bool beamBlocked,
  int gasValue,
  float temp,
  const char* state
) {

  char payload[220];

  snprintf(
    payload,
    sizeof(payload),

    "{\"pir\":%d,"
    "\"ldr\":%d,"
    "\"beam_blocked\":%d,"
    "\"gas\":%d,"
    "\"temperature\":%.2f,"
    "\"state\":\"%s\"}",

    motion ? 1 : 0,
    ldrValue,
    beamBlocked ? 1 : 0,
    gasValue,
    temp,
    state
  );

  client.publish(mqttTopic, payload);

  Serial.print("MQTT -> ");
  Serial.println(payload);
}

// =====================================================
// SETUP
// =====================================================
void setup() {

  Serial.begin(115200);

  pinMode(PIR_PIN, INPUT);
  pinMode(MQ_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();

  delay(2000);

  // Create unique ID from ESP32 chip
  uint64_t chipID = ESP.getEfuseMac();

  snprintf(
    clientId,
    sizeof(clientId),
    "FarmGuard-%04X",
    (uint16_t)(chipID >> 32)
  );

  snprintf(
    mqttTopic,
    sizeof(mqttTopic),
    "farmguard/%04X/data",
    (uint16_t)(chipID >> 32)
  );

  // WiFi
  connectWiFi();

  // MQTT
  client.setServer(mqtt_server, mqtt_port);

  connectMQTT();

  // Read initial sensor states
  int initialLdr = analogRead(LDR_PIN);

  previousBeamBlocked =
    (initialLdr < LDR_THRESHOLD);

  previousMotion =
    digitalRead(PIR_PIN);

  Serial.println();
  Serial.println("=================================");
  Serial.println("       FARMGUARD-AI ONLINE");
  Serial.println("=================================");

  Serial.print("MQTT Topic: ");
  Serial.println(mqttTopic);

  Serial.println("=================================");
}

// =====================================================
// LOOP
// =====================================================
void loop() {

  unsigned long now = millis();

  // =================================================
  // CONNECTION CHECK
  // =================================================

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (!client.connected()) {
    connectMQTT();
  }

  client.loop();

  // =================================================
  // READ SENSORS
  // =================================================

  bool motion = digitalRead(PIR_PIN);

  int ldrValue = analogRead(LDR_PIN);

  int gasValue = analogRead(MQ_PIN);

  bool beamBlocked =
    (ldrValue < LDR_THRESHOLD);

  // =================================================
  // TEMPERATURE
  // =================================================

  if (now - lastDhtTime >= DHT_INTERVAL) {

    lastDhtTime = now;

    float newTemperature =
      dht.readTemperature();

    if (!isnan(newTemperature)) {

      temperature = newTemperature;
    }
  }

  // =================================================
  // PIR EVENT
  // =================================================

  if (motion && !previousMotion) {

    lastPirEvent = now;

    Serial.println(">> PIR EVENT DETECTED");

    // Was laser broken recently?
    if (
      lastBeamEvent > 0 &&
      now - lastBeamEvent <= FUSION_WINDOW
    ) {

      confirmedUntil =
        now + ALERT_HOLD_TIME;

      Serial.println(
        ">> SENSOR FUSION MATCH!"
      );

      Serial.println(
        ">> INTRUSION CONFIRMED!"
      );
    }
  }

  // =================================================
  // LASER EVENT
  // =================================================

  if (
    beamBlocked &&
    !previousBeamBlocked
  ) {

    lastBeamEvent = now;

    Serial.println(
      ">> LASER BEAM BROKEN"
    );

    // Was PIR triggered recently?
    if (
      lastPirEvent > 0 &&
      now - lastPirEvent <= FUSION_WINDOW
    ) {

      confirmedUntil =
        now + ALERT_HOLD_TIME;

      Serial.println(
        ">> SENSOR FUSION MATCH!"
      );

      Serial.println(
        ">> INTRUSION CONFIRMED!"
      );
    }
  }

  previousMotion = motion;
  previousBeamBlocked = beamBlocked;

  // =================================================
  // DETERMINE SYSTEM STATE
  // =================================================

  bool confirmed =
    confirmedUntil > 0 &&
    ((long)(confirmedUntil - now) > 0);

  bool recentPir =
    lastPirEvent > 0 &&
    now - lastPirEvent <= FUSION_WINDOW;

  bool recentBeam =
    lastBeamEvent > 0 &&
    now - lastBeamEvent <= FUSION_WINDOW;

  const char* state;

  // CONFIRMED
  if (confirmed) {

    state = "CONFIRMED";

    digitalWrite(
      BUZZER_PIN,
      HIGH
    );
  }

  // WATCH
  else if (
    motion ||
    beamBlocked ||
    recentPir ||
    recentBeam
  ) {

    state = "WATCH";

    digitalWrite(
      BUZZER_PIN,
      LOW
    );
  }

  // NORMAL
  else {

    state = "NORMAL";

    digitalWrite(
      BUZZER_PIN,
      LOW
    );
  }

  // =================================================
  // MQTT PUBLISH
  // =================================================

  bool stateChanged =
    previousState != String(state);

  // Publish every 2 sec
  // OR immediately if state changes

  if (
    now - lastMqttTime >= MQTT_INTERVAL ||
    stateChanged
  ) {

    lastMqttTime = now;

    publishData(
      motion,
      ldrValue,
      beamBlocked,
      gasValue,
      temperature,
      state
    );

    previousState = String(state);
  }

  delay(50);
}
