import cv2
import numpy as np
import os
import tensorflow as tf


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model.signatures["serving_default"]


def test(model, image_path, expected):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (125, 125))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    result = model(tf.constant(img_input))

    if isinstance(result, dict):
        result = result["output_0"]

    result = result.numpy()[0][0]

    threshold = 0.5
    predicted = 1 if result >= threshold else 0
    return predicted == expected


def main():
    print("Running test suite...")
    ORGANIC = 0
    RECYCLABLE = 1

    model = load_model("model/saved_model")

    test_dir = "model/dataset/TEST/"
    recyclable_dir = os.path.join(test_dir, "R")
    organic_dir = os.path.join(test_dir, "O")

    tests_passed = 0
    tests_failed = 0

    # Recyclable tests
    for file_name in os.listdir(recyclable_dir):
        image_path = os.path.join(recyclable_dir, file_name)
        if test(model, image_path, RECYCLABLE):
            tests_passed += 1
        else:
            tests_failed += 1

    # Organic tests
    for file_name in os.listdir(organic_dir):
        image_path = os.path.join(organic_dir, file_name)
        if test(model, image_path, ORGANIC):
            tests_passed += 1
        else:
            tests_failed += 1

    test_count = tests_passed + tests_failed
    accuracy = (1 - (tests_failed / test_count)) * 100
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Test Count: {test_count}")
    print(f"Model Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
