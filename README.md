# DataSet
Organic vs Recyclable dataset was obtained from [kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)

# This is for Iman school 10A3 STEAM 2024-2025


- 0 = Organic
- 1 = Recyclable


# TODO
## Option 1
    1. Capture an image from the OV7670 camera using Arduino.
    2. Send image data in small chunks over UART to the WiFi module (ESP8266).
    3. ESP8266 forwards the image over WiFi to Flask server.
    4. Flask server processes the image using ai and sends the answer.
### NOTE: The ESP8266 will receive image data over Serial and send it via WiFi to the Flask server.

## Option 2
    1. Capture an image from the OV7670 camera using ESP8266.
    2. forward the image over WIFI to Flask server.
    3. Flask server processes the image using ai and sends the answer.
