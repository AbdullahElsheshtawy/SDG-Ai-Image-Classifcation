import socket
import os
import logging
from PIL import Image


def send_images(host, port=5001, num_images=10, image_dir="model/dataset/TRAIN/O/"):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        logging.info(f"Connected to server at {host}:{port}")

        for i in range(1, num_images + 1):
            image_path = os.path.join(image_dir, f"O_{i}.jpg")

            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize((128, 128), Image.LANCZOS)
                img_bytes = img.tobytes()
                client_socket.sendall(img_bytes)

                response = client_socket.recv(1024)
                if response:
                    prediction = response.decode("utf-8")
                    logging.info(f"Recieved prediction for image {i}: {prediction}")
                else:
                    logging.error(f"No repsonse recieved for image {i}")
                logging.info(f"Sent Image {i}: {image_path} ({len(img_bytes)} bytes)")

            except FileNotFoundError:
                logging.error(f"Image not found: {image_path}")
            except OSError as e:
                logging.error(f"Image Error: {e}")
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")

    except Exception as e:
        logging.error(f"Error: {e}")

    finally:
        # Check if client_socket exists
        if "client_socket" in locals() and client_socket is not None:
            client_socket.close()
            logging.info("Connection closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    send_images(host="127.0.0.1", port=5001, num_images=1000)
