import tensorflow as tf
import numpy as np
import cv2

def test_tflite_model(tflite_file, image_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Cannot load image:", image_path)

    img_resized = cv2.resize(img, (360, 240))  # Your input size (width, height)

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    print(f"Output Details {output_details}")
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data  # Return the prediction

# Example usage:
tflite_file = "model.tflite"
image_path = "dataset/test/R/R_10000.jpg" # A test image
ORGANIC = False

prediction = test_tflite_model(tflite_file, image_path)


print("Raw Prediction:", prediction)

# Binary classification:
threshold = 0.5  # Adjust if needed
predicted_class = 1 if prediction >= threshold else 0  # Assuming prediction is a 1-element array
print(predicted_class)
# 0 = Organic
# 1 = Recyclable
if ORGANIC == True and predicted_class == 0:
    print("Correct Prediction: Organic")
elif ORGANIC == False and predicted_class == 1:
    print("Correct Prediction: Recyclable")
else:
    print("Incorrect Prediction")



import json

with open("class_names.json", "r") as f:
    class_names_dict = json.load(f)

# ... (after getting the prediction)

threshold = 0.5
predicted_class = 1 if prediction[0] >= threshold else 0

# Use the saved mapping to get the correct class name:
for class_name, index in class_names_dict.items():
    if index == predicted_class:
        predicted_class_name = class_name
        break

print("Predicted Class Name:", predicted_class_name)  # The correct class name