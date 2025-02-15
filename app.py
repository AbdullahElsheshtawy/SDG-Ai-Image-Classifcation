from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
import socket
import threading
import time
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

lock = threading.Lock()

model = tf.saved_model.load("model/saved_model")
model = model.signatures["serving_default"]


def processImage(imageData):
    """Process image with TensorFlow saved model"""
    try:
        image = Image.frombytes("RGB", (177, 144), imageData)
        assert image.size == (177, 144)

        inputData = np.array(image, dtype=np.float32) / 255.0
        inputData = np.expand_dims(inputData, axis=0)

        startTime = time.time()

        result = model(tf.constant(inputData))
        if isinstance(result, dict):
            result = result["output_0"]

        result = result.numpy().item()
        with lock:
            stats["lastInferenceTime"] = time.time() - startTime
            stats["lastPrediction"] = float(result)
            stats["totalImagesProcessed"] += 1
        prediction = "Recyclable" if result >= 0.5 else "Organic"
        logging.info(f"Image: {prediction}")
        return result

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
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
            with lock:
                stats["connectionStatus"] = f"Connected to {addr}"
            logging.info(f"Connected to Arduino at {addr}")

            processImageStream(clientSocket)
        except Exception as e:
            logging.error(f"Connection Error: {e}")
            with lock:
                stats["connectionStatus"] = "Error" + str(e)


def processImageStream(clientSocket: socket.socket):
    imageBuffer = bytearray()
    framesProcessed = 0
    startTime = time.time()

    try:
        while True:
            data = clientSocket.recv(4096)
            if not data:
                break
            imageBuffer.extend(data)

            while len(imageBuffer) >= 177 * 144 * 3:
                imageData = imageBuffer[: 177 * 144 * 3]
                imageBuffer = imageBuffer[177 * 144 * 3 :]

                processImage(imageData)
                framesProcessed += 1
                elapsedTime = time.time() - startTime

                if elapsedTime >= 1.0:
                    with lock:
                        stats["fps"] = framesProcessed / elapsedTime
                        framesProcessed = 0
                        startTime = time.time()

    except Exception as e:
        logging.error(f"Stream Processing Error: {e}")

    finally:
        clientSocket.close()
        with lock:
            stats["connectionStatus"] = "Disconnected"
        logging.info("Arduino Disconnected")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats")
def getStats():
    with lock:
        return jsonify(stats)


if __name__ == "__main__":
    arduinoThread = threading.Thread(target=handle_arduino_connection, daemon=True)
    arduinoThread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
