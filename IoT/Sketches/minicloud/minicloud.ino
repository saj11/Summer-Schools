
#include "arduino_secrets.h" 

#include <WiFi101.h>
#include <MQTT.h>
#include <ArduinoJson.h>

const char ssid[] = SECRET_SSID;
const char pass[] = SECRET_PASS;

WiFiClient net;
MQTTClient client;

unsigned long lastMillis = 0;

// these are needed to manage the JSON messages
StaticJsonBuffer<200> jsonBuffer;
JsonObject& root = jsonBuffer.createObject();
char JSONmessageBuffer[100];

// handle connection to the wifi and the mqtt server
void connect() {
  Serial.print("checking wifi...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }

  Serial.print("\nconnecting...");
  while (!client.connect("device1", SECRET_USERNAME, SECRET_PASSWORD )) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("\nconnected!");
  client.subscribe("/devices/group1/input");
}

// Handle received messages
void messageReceived(String &topic, String &payload) {
  Serial.println("incoming: " + topic + " - " + payload);
  // you can put an "if" here and (for example) turn on and off a lamp depending 
  // on the message
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, pass);

  // name of the MQTT Server
  client.begin(SECRET_MQTTSERVER, net);
  client.onMessage(messageReceived);

  connect();
}

void loop() {
  client.loop();

  if (!client.connected()) {
    connect();
  }

  // publish a message roughly every second.
  if (millis() - lastMillis > 1000) {
    lastMillis = millis();

    // temperature sensor on analog input 0
    float t = analogRead(A0) / 10.0f;
    root["temperature"] = t;
    root["counter"] = millis()/1000;
 
    //place the JSON object into a nice String
    root.printTo(JSONmessageBuffer, sizeof(JSONmessageBuffer));
    //publish the message
    client.publish("/devices/group1/output", JSONmessageBuffer);
  }
}
