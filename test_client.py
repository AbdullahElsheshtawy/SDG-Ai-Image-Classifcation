import socket
import numpy as np
from PIL import Image
import time

def sendImages(host, port=5001, numImages=10):
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.connect((host, port))

    try: 
        for i in range(numImages):
            imageBytes = Image.open(f'model/dataset/TRAIN/O/O_{numImages}.jpg').tobytes()
            clientSocket.sendall(imageBytes)
            print(f"Sent Image {i+1}")
            time.sleep(1)
    finally:
        clientSocket.close()


if __name__ == "__main__":
    sendImages(host='192.168.100.38', port=5001, numImages=10)
