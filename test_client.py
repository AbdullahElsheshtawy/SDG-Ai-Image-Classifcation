import socket
import struct
import time
import os


def send_images(host, port=5001, num_images=10, image_dir="model/dataset/TRAIN/O"):
    """Send test images to the server via TCP."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")

        for i in range(1, num_images + 1):
            image_path = os.path.join(image_dir, f"O_{i}.jpg")

            if not os.path.exists(image_path):
                print(f"Skipping: {image_path} not found.")
                continue

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_size = len(image_bytes)
            client_socket.sendall(struct.pack("<I", image_size))  # Send size first
            client_socket.sendall(image_bytes)  # Send image data

            print(f"Sent Image {i}: {image_path} ({image_size} bytes)")

            # Optional: Wait for server response
            try:
                response = client_socket.recv(1024).decode()
                print(f"Server Response: {response}")
            except socket.timeout:
                print("No response from server (timeout).")

            time.sleep(1)  # Simulate a delay between sends

    except Exception as e:
        print(f"Error: {e}")

    finally:
        client_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    send_images(host="192.168.100.38", port=5001, num_images=10)
