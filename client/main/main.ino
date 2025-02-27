#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <ESP8266HTTPClient.h>

const char *ssid = "STC";
const char *password = "ALLAHOAKBAR";
constexpr int serverPort = 5001;
constexpr int discoveryPort = 8888;
WiFiUDP udp{};
IPAddress serverIP{};
WiFiClient client{};
bool isServerFound = false;
bool isServerConnected = false;
constexpr int IMAGE_SIZE = 128 * 128 * 2;
uint8_t imageBuffer[IMAGE_SIZE] = { 0 };

enum Prediction : uint8_t {
  ORGANIC = 0,
  RECYCLABLE = 1,
};


void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Coonected to WiFi: ");
  Serial.println(WiFi.localIP());

  udp.begin(discoveryPort);
  Serial.println("Listening for server announcements");
}

void loop() {
  if (isServerFound) {
    if (isServerConnected) {
      SendImage();
      Prediction prediction = RecievePrediction();
      if (prediction == ORGANIC)
        Serial.println("Organic");
      else
        Serial.println("Recyclable");

    } else {
      ConnectToServer();
    }
  } else {
    FindServer();
    delay(100);
  }
}

void FindServer() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char packetBuffer[255];
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = '\0';

      Serial.print("Recieved packet: ");
      Serial.println(packetBuffer);

      if (String(packetBuffer).startsWith("SERVER_ANNOUNCE:")) {
        String ipStr = String(packetBuffer).substring(16);
        
        Serial.print("IP String: ");
        Serial.println(ipStr);

        if (serverIP.fromString(ipStr)) {
          isServerFound = true;
          Serial.print("Server discovered at: ");
          Serial.println(serverIP.toString());
        } else {
          Serial.println("IP conversion failed");
        }
      } else {
        Serial.println("Could not find Server announcement retrying");
      }
    }
  }
}

void ConnectToServer() {
  Serial.print("Connecting to server at: ");
  Serial.println(serverIP.toString() + ":" + String(serverPort));

  if (client.connect(serverIP, serverPort)) {
    Serial.println("Connected to server");
    isServerConnected = true;
  } else {
    Serial.println("Could not connect to server. Retrying discovery");
    isServerConnected = false;
    isServerFound = false;
  }
}


void SendImage() {
  client.write(imageBuffer, IMAGE_SIZE);
}

Prediction RecievePrediction() {
  while (!client.available()) delay(10);
  return static_cast<Prediction>(client.read());
}