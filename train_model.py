import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import json
import numpy as np

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

TRAIN_DIR = r"C:\Users\user\OneDrive\Desktop\dataset\PlantVillage\train"
VAL_DIR = r"C:\Users\user\OneDrive\Desktop\dataset\PlantVillage\val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 13
EPOCHS = 10

# --------------------------------------------------
# 2. CLASS NAMES
# --------------------------------------------------

class_names = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",

    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",

    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",

    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
]

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Loading validation dataset...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Dataset loaded successfully.")

# --------------------------------------------------
# 4. OPTIMIZE DATA PIPELINE
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# --------------------------------------------------
# 5. DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# --------------------------------------------------
# 6. LOAD MOBILE NET V2
# --------------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pre-trained model
base_model.trainable = False

# --------------------------------------------------
# 7. BUILD OUR MODEL
# --------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# --------------------------------------------------
# 8. COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 9. DISPLAY MODEL
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# 10. TRAIN MODEL
# --------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

model.save("crop_disease_model.keras")

# --------------------------------------------------
# 12. SAVE CLASS NAMES
# --------------------------------------------------

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("\nTraining completed!")
print("Model saved as: crop_disease_model.keras")
print("Class names saved as: class_names.json")