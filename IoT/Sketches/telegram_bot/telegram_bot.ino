/*
  EchoBot

  This example shows how to program a Telegram Bot
  that echoes your messages using a custom keyboard.

  For a step-by-step tutorial visit:
  https://create.arduino.cc/projecthub/Arduino_Genuino/telegram-bot-library-ced4d4

  In oreder to make the bot more reliable in the long run we suggest using a watchdog
  The Adafruit_SleepyDog is a handy library that will reset the board if something goes wrong

  Updated 29 May 2016
  by Tommaso Laterza
  Updated 13 February 2018
  by Tommaso Laterza

  This example code is in the public domain.

*/

#include "arduino_secrets.h"
#include <WiFi101.h>
#include <SPI.h>
#include <TelegramBot.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// Initialize Wifi connection to the router
char ssid[] = SECRET_SSID;             // your network SSID (name)
char pass[] = SECRET_PASS;           // your network key
char pass2[] = SECRET_WPWD;        // your network password (use for WPA, or use as key for WEP)
char apikey[] = SECRET_APIKEY;    // a key used to query OpenWeatherMap
char weatherId[10];
char server[] = "api.openweathermap.org";

const unsigned long HTTP_TIMEOUT = 10000; 

// Initialize Telegram BOT
const char BotToken[] = SECRET_BOT_TOKEN;

String city = "San Jose,cr";

WiFiSSLClient client;
TelegramBot bot (BotToken, client);
TelegramKeyboard keyboard_one;

int const redLEDPin = A1;
int const greenLEDPin = A3;
int const blueLEDPin = A2;
int keyIndex = 0; 
int status = WL_IDLE_STATUS;

WiFiClient client;

void setup() {
  lcd.begin(16, 1);
  
  pinMode(greenLEDPin, OUTPUT);
  pinMode(redLEDPin, OUTPUT);
  pinMode(blueLEDPin, OUTPUT);

  Serial.begin(115200);
  while (!Serial) {} // Wait for the Serial monitor to be opened
  delay(3000);

  // attempt to connect to Wifi network:
  Serial.print("Connecting Wifi: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("");
  lcd.clear();
  lcd.print("Iniciando...");
  Serial.println("WiFi connected");

  // define your row's
  const char* row_one[] = {"true", "false"};
  const char* row_two[] = {"red", "gree", "blue", "hello!"};

  // assing a row to one or more keyboards
  // second argument is the length of the row
  keyboard_one.addRow(row_one, 2);
  keyboard_one.addRow(row_two, 4);
  bot.begin();
}

void loop() {
  performHttpRequest();
  
  double intensidad = 100;
  message m = bot.getUpdates(); // Read new messages
  if ( m.chat_id != 0 ) { // Checks if there are some updates
    lcd.clear();
    Serial.println(m.text);
    lcd.print(m.text);
    //bot.sendMessage(m.chat_id, m.text, keyboard_one);  // Reply to the same chat with the same text
    intensidad = (m.text.substring(m.text.lastIndexOf(" "))).toInt();
    Serial.println(intensidad/100);
    Serial.println(m.text.substring(0,m.text.lastIndexOf(" ")));
    if(m.text.substring(0,m.text.lastIndexOf(" ")) == "red"){
        setColor(255*(intensidad/100),0,0);
    }else if(m.text.substring(0,m.text.lastIndexOf(" ")) == "green"){
        setColor(0,255*(intensidad/100),0);
    }else if(m.text.substring(0,m.text.lastIndexOf(" ")) == "blue"){
        setColor(0,0,255*(intensidad/100));
    }
  } else {
    Serial.println("no new message");
  }

  delay(2000);
}

void setColor(int red, int green, int blue)
{
  #ifdef COMMON_ANODE
    red = 255 - red;
    green = 255 - green;
    blue = 255 - blue;
  #endif
  Serial.println(red);
  Serial.println(green);
  Serial.println(blue);
  analogWrite(redLEDPin, red);
  analogWrite(greenLEDPin, green);
  analogWrite(blueLEDPin, blue);  
}

void performHttpRequest() {
  
    Serial.println("\nStarting connection to server...");
  // if you get a connection, report back via serial:
  if (client.connect(server, 80)) {
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
