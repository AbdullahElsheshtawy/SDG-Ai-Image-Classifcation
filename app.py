import tensorflow as tf
import numpy as np
import asyncio
import logging
from PIL import Image
import time
from quart import Quart, jsonify, render_template
from collections import defaultdict
from hypercorn.asyncio import Config, serve
import socket

app = Quart(__name__)

IMAGE_SIZE = 128
IMAGE_SIZE_IN_BYTES = IMAGE_SIZE * IMAGE_SIZE * 2


class Model:
    def __init__(self):
        model = tf.saved_model.load("model/saved_model")
        model = model.signatures["serving_default"]
        self.model = model

    def Predict(self, inputTensor):
        return self.model(inputTensor)


model = Model()


class Stats:
    def __init__(self):
        self.totalImagesProcessed = 0
        self.lastInferenceTime = 0.0
        self.lastPrediction = "-"
        self.fps = 0.0
        self.framesProcessed = 0
        self.activeConnections = 0
        self.connectionDetails = defaultdict(
            lambda: {"lastInferenceTime": 0.0, "lastPrediction": "-"}
        )
        self.lock = asyncio.Lock()

    async def FinishedProcessingImage(
        self,
        clientId: str,
        inferenceTime: float,
        prediction: str,
    ):
        async with self.lock:
            self.totalImagesProcessed += 1
            self.lastInferenceTime = inferenceTime
            self.lastPrediction = prediction
            self.connectionDetails[clientId] = {
                "lastInferenceTime": inferenceTime,
                "lastPrediction": prediction,
            }
            self.framesProcessed += 1

    async def Connected(self, clientId: str):
        async with self.lock:
            self.activeConnections += 1
            if clientId not in self.connectionDetails:
                self.connectionDetails[clientId] = {
                    "lastInferenceTime": 0.0,
                    "lastPrediction": "-",
                }

    async def Disconnected(self, clientId: str):
        async with self.lock:
            self.activeConnections -= 1
            if clientId in self.connectionDetails:
                del self.connectionDetails[clientId]

    async def GetStats(self):
        return {
            "totalImagesProcessed": self.totalImagesProcessed,
            "activeConnections": self.activeConnections,
            "FPS": self.fps,
            "lastPredicted": self.lastPrediction,
            "inferenceTime": self.lastInferenceTime,
            "connectionDetails": self.connectionDetails,
        }


stats = Stats()


async def fps_updater():
    while True:
        await asyncio.sleep(1.0)
        async with stats.lock:
            # Capture the count and reset
            current_frames = stats.framesProcessed
            stats.fps = current_frames
            stats.framesProcessed = 0


def RGB565ToRGB8(imageData: bytes) -> bytes:
    arr = np.frombuffer(imageData, dtype=np.uint16)  # Load as uint16 array
    r = (arr >> 11) & 0x1F
    g = (arr >> 5) & 0x3F
    b = arr & 0x1F

    r = (r << 3).astype(np.uint8)  # Convert 5-bit to 8-bit
    g = (g << 2).astype(np.uint8)  # Convert 6-bit to 8-bit
    b = (b << 3).astype(np.uint8)  # Convert 5-bit to 8-bit

    return np.stack((r, g, b), axis=-1)  # Stack channels to form an RGB image


async def processImage(imageData: bytes, clientId) -> bytes:
    try:
        rgb565Data = RGB565ToRGB8(imageData)
        image = Image.frombytes("RGB", (IMAGE_SIZE, IMAGE_SIZE), rgb565Data)
        assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
        inputData = (
            np.frombuffer(image.tobytes(), dtype=np.uint8)
            .reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
            .astype(np.float32)
            / 255.0
        )
        inputData = np.expand_dims(inputData, axis=0)

        startTime = time.time()
        result = model.Predict(tf.constant(inputData))

        if isinstance(result, dict):
            result = result["output_0"]
        result: float = result.numpy().item()

        inferenceTime = (time.time() - startTime) * 1000
        prediction = "Recyclable" if result >= 0.5 else "Organic"

        await stats.FinishedProcessingImage(clientId, inferenceTime, prediction)
        return b"\x01" if result > 0.5 else b"\x00"
    except Exception as e:
        logging.error(f"Error Processing image: {e}")
        return None


async def HandleClient(reader, writer):
    addr = writer.get_extra_info("peername")
    clientId = f"{addr[0]}:{addr[1]}"

    sock: socket.socket = writer.get_extra_info('socket')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

    imageBuffer = bytearray()
    startTime = time.time()

    await stats.Connected(clientId)

    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break

            imageBuffer.extend(data)

            while len(imageBuffer) >= IMAGE_SIZE_IN_BYTES:
                imageData = imageBuffer[:IMAGE_SIZE_IN_BYTES]
                imageBuffer = imageBuffer[IMAGE_SIZE_IN_BYTES:]

                prediction = await processImage(imageData, clientId)
                writer.write(prediction)
                await writer.drain()
                elapsedTime = time.time() - startTime

                if elapsedTime >= 1.0:
                    startTime = time.time()
    except (ConnectionResetError, asyncio.CancelledError, OSError, ConnectionError) as e:
        logging.info(f"Connection with {clientId} was closed: {e}")
    except Exception as e:
        logging.info(f"Error handling client: {e}")

    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logging.info(f"Closing connection with {clientId}: {e}")

        await stats.Disconnected(clientId)


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


async def BroadcastIP():
    UDP_IP = "255.255.255.255"
    UDP_PORT = 8888
    SERVER_PORT = 5001

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    serverIP = socket.gethostbyname(socket.gethostname())

    message = f"SERVER_ANNOUNCE:{serverIP}".encode()

    while True:
        sock.sendto(message, (UDP_IP, UDP_PORT))
        logging.info(f"Broadcasted server IP: {serverIP}")
        await asyncio.sleep(5)


@app.route("/")
async def Index():
    return await render_template("index.html")


@app.route("/stats")
async def GetStats():
    return jsonify(await stats.GetStats())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    @app.before_serving
    async def Startup():
        app.tcp_server = asyncio.create_task(RunServer())
        app.fps_task = asyncio.create_task(fps_updater())
        app.broadcast_task = asyncio.create_task(BroadcastIP())

    @app.after_serving
    async def Shutdown():
        if hasattr(app, "tcp_server"):
            app.tcp_server.cancel()
            app.fps_task.cancel()
            app.broadcast_task.cancel()
            try:
                await app.tcp_server
                await app.fps_task
                await app.broadcast_task
            except asyncio.CancelledError:
                pass

    config = Config()
    config.bind = ["0.0.0.0:5000"]

    asyncio.run(serve(app, config))
