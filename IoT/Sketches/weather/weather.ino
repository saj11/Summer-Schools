// web client which downloads the weather for a certain city

#include "arduino_secrets.h"
#include <ArduinoJson.h>
#include <ESP8266WiFi.h> 
///////please enter your sensitive data in the Secret tab/arduino_secrets.h
const char* ssid     = "Joseph";
const char* password = "future11$";
char apikey[] = "6fc633ea289c69e73dc7fd7a13ccb5eb";    // a key used to query OpenWeatherMap
String city = "San Jose,cr";

char weatherId[10];

const unsigned long HTTP_TIMEOUT = 10000; 

// the HTTP request we make to the server
String request = "GET /data/2.5/weather?q=" + city + "&appid=" + SECRET_APIKEY +" HTTP/1.1";

int keyIndex = 0;            // your network key Index number (needed only for WEP)

int status = WL_IDLE_STATUS;
// if you don't want to use DNS (and reduce your sketch size)
// use the numeric IP instead of the name for the server:
//IPAddress server(74,125,232,128);  // 
const char* host = "api.openweathermap.org";

// Initialize the Ethernet client library
// with the IP address and port of the server
// that you want to connect to (port 80 is default for HTTP):
WiFiClient client;

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // check for the presence of the shield:
  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    // don't continue:
    while (true);
  }

  // attempt to connect to WiFi network:
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    status = WiFi.begin(ssid, password);

    // wait 10 seconds for connection:
    delay(10000);
  }
  Serial.println("Connected to wifi");
  printWiFiStatus();


}

void loop() {
  // if there are incoming bytes available
  // from the server, read them and print them:
  performHttpRequest();
/*  while (client.available()) {
    char c = client.read();
    Serial.write(c);
  }*/

  // if the server's disconnected, stop the client:
  if (!client.connected()) {
    Serial.println();
    Serial.println("disconnecting from server.");
    client.stop();


  }
  delay(10000);
}


void performHttpRequest() {
  
    Serial.println("\nStarting connection to server...");
  // if you get a connection, report back via serial:
  if (client.connect(host, 80)) {
    Serial.println("connected to server");
    // Make a HTTP request:
    client.println(request);
    client.println("Host: api.openweathermap.org");
    client.println("Connection: close");
    client.println();
    if (skipResponseHeaders() ) {
      Serial.println("parsing data");
 
      DynamicJsonBuffer jsonBuffer(2048);
        
      JsonObject& root = jsonBuffer.parseObject(client);
      root.prettyPrintTo(Serial);
      Serial.println();

      if (!root.success()) {
        Serial.println("JSON parsing failed!");
      } else {
        
        int    w = root["weather"][0]["id"];
        String m = root["weather"][0]["main"];
        String d = root["weather"][0]["description"];
        
        Serial.print("The Weather in ");
        Serial.print(city);
        Serial.print(" is ");
        Serial.print(m);
        Serial.print(" (");
        Serial.print(d);
        Serial.println(")");
      }
    }
  }
  
  
  
}


// Skip HTTP headers so that we are at the beginning of the response's body
bool skipResponseHeaders() {
  // HTTP headers end with an empty line
  char endOfHeaders[] = "\r\n\r\n";

  client.setTimeout(HTTP_TIMEOUT);
  bool ok = client.find(endOfHeaders);

  if (!ok) {
    Serial.println("No response or invalid response!");
  }

  return ok;
}

void printWiFiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
