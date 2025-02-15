import os
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision


def lr_schedule(epoch):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 5e-4
    else:
        return 1e-4


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


train_dir = "model/dataset/TRAIN"
test_dir = "model/dataset/TEST"

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


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    rescale=1.0 / 255,
    validation_split=0.2,
)


# No augmentation on test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 64
img_size = (128, 128)

train_set = train_datagen.flow_from_directory(
    train_dir,
    class_mode="binary",
    batch_size=batch_size,
    target_size=img_size,
    subset="training",
    shuffle=True,
)


validation_set = train_datagen.flow_from_directory(
    train_dir,
    class_mode="binary",
    batch_size=batch_size,
    target_size=img_size,
    subset="validation",
)

test_set = test_datagen.flow_from_directory(
    test_dir, class_mode="binary", batch_size=batch_size, target_size=img_size
)

img_shape = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape, include_top=False, weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", AUC()]
)

steps_per_epoch = len(train_set)
validation_steps = len(validation_set)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [early_stopping, lr_scheduler]

history2 = model.fit(
    train_set,
    epochs=20,
    validation_data=validation_set,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=callbacks,
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

finetune_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    optimizer=finetune_optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", AUC()],
)

history_finetune = model.fit(
    train_set,
    epochs=10,
    validation_data=validation_set,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=callbacks,
)

plot_graphs(history2, "accuracy")
plot_graphs(history2, "loss")
plot_graphs(history_finetune, "accuracy")
plot_graphs(history_finetune, "loss")

results = model.evaluate(test_set, verbose=True)
print(f"Results: {results}")
tf.saved_model.save(model, export_dir="model/saved_model")
print("Finished")
