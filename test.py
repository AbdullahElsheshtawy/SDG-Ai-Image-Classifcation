import cv2
import numpy as np
import os
import tensorflow as tf  

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def test(image_path, expected):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (360, 240))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # Set the input tensore
    interpreter.set_tensor(input_details[0]["index"], img_input)
    interpreter.invoke()

    # 0..1
    result = interpreter.get_tensor(output_details[0]['index'])

    # 0 = Organic
    # 1 = Recyclable
    threshold = 0.5
    predicted = 1 if result >= threshold else 0
    return predicted == expected
        
    
def main():
    print("Running test suite...")
    ORGANIC = 0
    RECYCLABLE = 1
    test_dir = "dataset/TEST/"
    recyclable_dir = os.path.join(test_dir, "R")
    organic_dir = os.path.join(test_dir, "O")  

    tests_passed = 0
    tests_failed = 0
    # Recyclable tests
    for file_name in os.listdir(recyclable_dir):
        image_path = os.path.join(recyclable_dir, file_name)
        if test(image_path, RECYCLABLE):
            tests_passed += 1
        else:
            tests_failed += 1
            

    # Organic tests
    for file_name in os.listdir(organic_dir):
        image_path = os.path.join(organic_dir, file_name)
        if test(image_path, ORGANIC):
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

