from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "crop_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully.")

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Save uploaded image
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    # Open image
    image = Image.open(file_path).convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to numpy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)

    # Find highest probability
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    # Separate crop and disease
    parts = predicted_class.split("___")

    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else predicted_class

    return render_template(
        "result.html",
        image_path=file.filename,
        crop=crop,
        disease=disease,
        confidence=round(confidence, 2)
    )

# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)