import tensorflow as tf
import numpy as np
import json

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paths
VAL_DIR = r"C:\Users\user\OneDrive\Desktop\dataset\PlantVillage\val"
MODEL_PATH = "crop_disease_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained model
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(val_dataset)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Get predictions efficiently
print("\nGenerating predictions...")
predictions = model.predict(val_dataset)
y_pred = np.argmax(predictions, axis=1)

# Extract true labels
y_true = np.concatenate([labels.numpy() for _, labels in val_dataset])

# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(14, 12))

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title("Crop Disease Classification - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as confusion_matrix.png")