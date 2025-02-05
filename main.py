import os
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def convert():
    model = tf.keras.models.load_model("model.keras") 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimizations (quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)


train_dir = "dataset/TRAIN"
test_dir = "dataset/TEST"

# 0 = Organic
# 1 = Recyclable
class_names = os.listdir((train_dir))

count_train = {}
count_tests = {}

for name in class_names:
    count_train[name] = len(os.listdir(os.path.join(train_dir, name)))
    count_tests[name] = len(os.listdir(os.path.join(test_dir, name)))

print(f"Images in train set: {sum(list(count_train.values()))}")
print(f"Images in tests set: {sum(list(count_tests.values()))}")



train_datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip = True,
                                   rescale=1./255,
                                   validation_split=0.2)


test_datagen = ImageDataGenerator(rescale = 1./255) # No augmentation on test set


batch_size = 32
img_size = (240, 360)

train_set = train_datagen.flow_from_directory(train_dir, 
                                              class_mode='binary',
                                              batch_size = batch_size,
                                              target_size=img_size,
                                              subset='training')

validation_set = train_datagen.flow_from_directory(train_dir, 
                                              class_mode='binary',
                                              batch_size = batch_size,
                                              target_size=img_size,
                                              subset='validation')

test_set = test_datagen.flow_from_directory(test_dir, 
                                           class_mode = 'binary',
                                           batch_size = batch_size, 
                                           target_size=img_size)

import json

class_names_dict = train_set.class_indices
with open("class_names.json", "w") as f:
    json.dump(class_names_dict, f)

print("Class Indices (Saved):", class_names_dict)


img_shape = (240, 360, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False, 
                                               weights='imagenet')

base_model.trainable = False  # Freeze initial layers

model = tf.keras.Sequential([base_model,
                             GlobalAveragePooling2D(),  # Corrected: Global Average Pooling
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(1, activation="sigmoid")                                     
                             ])

model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

steps_per_epoch=len(train_set) // batch_size
validation_steps = len(validation_set) // batch_size

history2 = model.fit(
    train_set,
    epochs = 30, # Increased epochs
    validation_data = validation_set, # Corrected validation data
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)

plot_graphs(history2, "accuracy")
plot_graphs(history2, "loss")

loss, accuracy = model.evaluate(test_set, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save("model.keras")

convert()

print("Finished")