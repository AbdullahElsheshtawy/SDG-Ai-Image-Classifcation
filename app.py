from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
import socket
import threading
import time
import io
import struct
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

stats = {
    "totalImagesProcessed": 0,
    "lastInferenceTime": 0,
    "lastPrediction": None,
    "connectionStatus": "Disconnected",
    "fps": 0,
}


interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()


def processImage(imageData):
    """Process image with Tensorflowlite model"""
    try:
        image = Image.open(io.BytesIO(imageData))

        # Only here for local testing. Remove if with arduino (arduino sends the image at the correct resolution)
        image = image.resize((360, 240))

        inputData = np.array(image, dtype=np.float32) / 255.0
        inputData = np.expand_dims(inputData, axis=0)

        startTime = time.time()

        interpreter.set_tensor(inputDetails[0]["index"], inputData)
        interpreter.invoke()
        outputData = interpreter.get_tensor(outputDetails[0]["index"])

        stats["lastInferenceTime"] = time.time() - startTime
        stats["lastPrediction"] = float(outputData[0][0])
        stats["totalImagesProcessed"] += 1
        prediction = "Recyclable" if outputData >= 0.5 else "Organic"
        logging.info(f"Image: {prediction}")
        return outputData

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None


def handle_arduino_connection(host="0.0.0.0", port=5001):
    """Handle TCP Connection From Arduino"""

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind((host, port))
    serverSocket.listen(1)
    logging.info(f"Listening for Arduino connection on port {port}")

    while True:
        try:
            clientSocket, addr = serverSocket.accept()
            stats["connectionStatus"] = f"Connected to {addr}"
            logging.info(f"Connected to Arduino at {addr}")

            processImageStream(clientSocket)
        except Exception as e:
            logging.error(f"Connection Error: {e}")
            stats["connectionStatus"] = "Error" + str(e)
            time.sleep(1)


def processImageStream(clientSocket: socket.socket):
    """Receives and processes images from the Arduino."""
    imageBuffer = bytearray()
    framesProcessed = 0
    startTime = time.time()

    try:
        while True:
            data = clientSocket.recv(4096)
            if not data:
                break

            imageBuffer.extend(data)

            while len(imageBuffer) >= 4:
                imageSize = struct.unpack("<I", imageBuffer[:4])[0]
                if len(imageBuffer) < imageSize + 4:
                    break

                imageData = imageBuffer[4 : imageSize + 4]
                imageBuffer = imageBuffer[imageSize + 4 :]

                processImage(imageData)
                framesProcessed += 1

                elapsedTime = time.time() - startTime
                if elapsedTime >= 1.0:
                    stats["fps"] = framesProcessed / elapsedTime
                    framesProcessed = 0
                    startTime = time.time()
            if len(imageBuffer) > 1024 * 1024 * 1024:  # If buffer grows too large
                logging.warning("Buffer overflow risk! Clearing buffer.")
                imageBuffer.clear()

    except Exception as e:
        logging.error(f"Stream Processing Error: {e}")

    finally:
        clientSocket.close()
        stats["connectionStatus"] = "Disconnected"
        logging.info("Arduino Disconnected")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats")
def getStats():
    return jsonify(stats)


if __name__ == "__main__":
    arduinoThread = threading.Thread(target=handle_arduino_connection)
    arduinoThread.start()

    app.run(host="0.0.0.0", port=5000, debug=True)
