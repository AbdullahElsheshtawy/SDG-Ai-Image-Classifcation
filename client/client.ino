#include <WiFi.h>
#include <esp_camera.h>
#include <esp_psram.h>
#include <WiFiUdp.h>
#include <esp_http_client.h>

constexpr auto PWDN_GPIO_NUM = 32;
constexpr auto RESET_GPIO_NUM = -1;
constexpr auto XCLK_GPIO_NUM = 0;
constexpr auto SIOD_GPIO_NUM = 26;
constexpr auto SIOC_GPIO_NUM = 27;
constexpr auto Y9_GPIO_NUM = 35;
constexpr auto Y8_GPIO_NUM = 34;
constexpr auto Y7_GPIO_NUM = 39;
constexpr auto Y6_GPIO_NUM = 36;
constexpr auto Y5_GPIO_NUM = 21;
constexpr auto Y4_GPIO_NUM = 19;
constexpr auto Y3_GPIO_NUM = 18;
constexpr auto Y2_GPIO_NUM = 5;
constexpr auto VSYNC_GPIO_NUM = 25;
constexpr auto HREF_GPIO_NUM = 23;
constexpr auto PCLK_GPIO_NUM = 22;
constexpr auto FLASH_LED_GPIO_NUM = 4;

constexpr int SERVER_PORT = 5001;
constexpr int DISCOVERY_PORT = 8888;
constexpr int IMAGE_WIDTH = 128;
constexpr int IMAGE_HEIGHT = 128;
constexpr int IMAGE_SIZE_IN_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * 2;

const char* ssid = "STC";
const char* password = "ALLAHOAKBAR";

WiFiUDP udp{};
IPAddress server_ip{};
WiFiClient client{};
bool found_server = false;
bool connected_to_server = false;


enum class Prediction : uint8_t {
  Organic = 0,
  Recyclable = 1,
};

const char* prediction_to_string(const Prediction prediction) {
  if (prediction == Prediction::Organic) {
    return "INFO: Image is Organic";
  } else if (prediction == Prediction::Recyclable) {
    return "Recyclable";
  } else {
    return "UNDEFINED";
  }
}


constexpr camera_config_t CAMERA_CONFIG = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sccb_sda = SIOD_GPIO_NUM,
  .pin_sccb_scl = SIOC_GPIO_NUM,

  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,
  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,

  // XCLK 20MHz or 10MHz for OV2640 double FPS (expiremental)
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,

  .pixel_format = PIXFORMAT_RGB565,
  .frame_size = FRAMESIZE_128X128,
  .fb_count = 1,
  .fb_location = CAMERA_FB_IN_PSRAM,

  .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  esp_camera_init(&CAMERA_CONFIG);
  wifi_connect();
  udp.begin(DISCOVERY_PORT);
  Serial.println("INFO: Listening for server announcements");
}

void loop() {
  if (!found_server) {
    found_server = find_server();
    Serial.printf("INFO: %s\n", found_server ? "Found server" : "Could not find server");
    delay(1000);
    return;
  }
  if (!connected_to_server) {
    connected_to_server = connect_to_server();
    Serial.printf("INFO: %s\n", connected_to_server ? "Connected to Server" : "Could not connect to server");
    delay(1000);
    return;
  }
  capture_and_send_image();
  const Prediction prediction = recieve_prediction();
  Serial.printf("INFO: Image is %s\n", prediction_to_string(prediction));
}

void wifi_connect() {
  WiFi.begin(ssid, password);
  Serial.print("INFO: WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("INFO: WiFi connected: ");
  Serial.println(WiFi.localIP());
}

bool find_server() {
  const int packet_size = udp.parsePacket();
  if (packet_size) {
    char packet_buffer[255]{};
    const int len = udp.read(packet_buffer, 255);

    if (len > 0) {
      packet_buffer[len] = '\0';

      Serial.printf("INFO: Recieved Packet: %s\n", packet_buffer);

      if (const String ip_string(packet_buffer); ip_string.startsWith("SERVER_ANNOUNCE:")) {
        Serial.printf("INFO: IP String: %s\n", ip_string.c_str());

        if (server_ip.fromString(ip_string)) {
          Serial.printf("INFO: Server discovered at: %s\n", ip_string.c_str());
          return true;
        } else {
          Serial.println("ERROR: IP conversion failed");
        }
      } else {
        Serial.println("INFO: Could not find server announcement retrying");
      }
    }
  }
  return false;
}

bool connect_to_server() {
  Serial.printf("INFO: Connecting to server at: %s:%zu\n", server_ip.toString().c_str(), SERVER_PORT);

  if (client.connect(server_ip, SERVER_PORT)) {
    Serial.println("INFO: Connected to server");
    return true;
  }
  Serial.println("ERROR: Could not connect to server. Retrying discovery");
  return false;
}

void capture_and_send_image() {
  auto* frame_buffer = esp_camera_fb_get();
  if (frame_buffer->height * frame_buffer->width != IMAGE_WIDTH * IMAGE_HEIGHT) {
    Serial.printf("ERROR: Image size expected to be %zuX%zu but image is %%zuX%zu\n", IMAGE_WIDTH, IMAGE_HEIGHT, frame_buffer->width, frame_buffer->height);
  }
  client.write(frame_buffer->buf, IMAGE_SIZE_IN_BYTES);
  esp_camera_fb_return(frame_buffer);
}

Prediction recieve_prediction() {
  while (!client.available()) delay(10);
  uint8_t prediction = 0xff;
  client.read(&prediction, 1);
  return static_cast<Prediction>(prediction);
}