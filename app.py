import tensorflow as tf
import numpy as np
import asyncio
import logging
from PIL import Image
import time
from quart import Quart, jsonify, render_template
from collections import defaultdict

app = Quart(__name__)
model = tf.saved_model.load("model/saved_model")
model = model.signatures["serving_default"]

IMAGE_SIZE = 128
IMAGE_SIZE_IN_BYTES = IMAGE_SIZE * IMAGE_SIZE * 3


class Stats:
    def __init__(self):
        self._data = {
            "totalImagesProcessed": 0,
            "lastInferenceTime": defaultdict(float),
            "lastPrediction": {},
            "connections": {},
            "fps": defaultdict(float),
        }
        self._lock = asyncio.Lock()

    async def Update(self, clientId, **kwargs):
        async with self._lock:
            for key, value in kwargs.items():
                if key in self._data:
                    if isinstance(self._data[key], dict):
                        self._data[key][clientId] = value
                    else:
                        self._data[key] += value

    async def GetStats(self):
        async with self._lock:
            activeConnections = len(
                [v for v in self._data["connections"].values() if v == "Connected"]
            )
            avgFps = sum(self._data["fps"].values()) / max(activeConnections, 1)
            return {
                "totalImagesProcessed": self._data["totalImagesProcessed"],
                "activeConnections": activeConnections,
                "connectionDetails": dict(self._data["connections"]),
                "averageFPS": round(avgFps, 2),
                "lastPredictions": dict(self._data["lastPrediction"]),
                "lastInferenceTimes": {
                    k: round(v, 3) for k, v in self._data["lastInferenceTime"].items()
                },
            }


stats = Stats()


async def processImage(imageData, clientId):
    try:
        image = Image.frombytes("RGB", (IMAGE_SIZE, IMAGE_SIZE), imageData)
        assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
        inputData = np.array(image, dtype=np.float32) / 255.0
        inputData = np.expand_dims(inputData, axis=0)

        startTime = time.time()
        result = model(tf.constant(inputData))

        if isinstance(result, dict):
            result = result["output_0"]
        result = result.numpy().item()

        inferenceTime = time.time() - startTime
        prediction = "Recyclable" if result >= 0.5 else "Organic"

        await stats.Update(
            clientId,
            totalImagesProcessed=1,
            lastInferenceTime=inferenceTime,
            lastPrediction=prediction,
        )

        logging.info(f"Client {clientId}: {prediction}")
        return result
    except Exception as e:
        logging.error(f"Error Processing image: {e}")
        return None


async def HandleClient(reader, writer):
    addr = writer.get_extra_info("peername")
    clientId = f"{addr[0]}:{addr[1]}"

    imageBuffer = bytearray()
    framesProcessed = 0
    startTime = time.time()

    logging.info(f"New connection from {clientId}")
    await stats.Update(clientId, connections={clientId: "Connected"})

    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break

            imageBuffer.extend(data)

            while len(imageBuffer) >= IMAGE_SIZE_IN_BYTES:
                imageData = imageBuffer[:IMAGE_SIZE_IN_BYTES]
                imageBuffer = imageBuffer[IMAGE_SIZE_IN_BYTES:]

                await processImage(imageData, clientId)

                framesProcessed += 1
                elapsedTime = time.time() - startTime

                if elapsedTime >= 1.0:
                    await stats.Update(clientId, fps=framesProcessed / elapsedTime)
                    framesProcessed = 0
                    startTime = time.time()

    except Exception as e:
        logging.error(f"Error handling client: {e}")

    finally:
        writer.close()
        await writer.wait_closed()
        await stats.Update(clientId, connections={clientId: "Disconnected"})
        logging.info(f"Client {clientId}: Disconnected")


async def RunServer():
    server = await asyncio.start_server(
        HandleClient,
        "0.0.0.0",
        5001,
        backlog=100,
    )

    addr = server.sockets[0].getsockname()
    logging.info(f"Serving on: {addr}")

    async with server:
        await server.serve_forever()


@app.route("/")
async def Index():
    return await render_template("index.html")


@app.route("/stats")
async def GetStats():
    return jsonify(await stats.GetStats())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(RunServer())

    app.run(host="0.0.0.0", port=5000, use_reloader=True)
