from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = load_model("cancer_detection_model.h5")

# Function to preprocess the input image
def preprocess_image(image, target_size=(48, 48)):
    # Convert the image to RGB format
    if image.shape[2] == 4:  # If the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.resize(image, target_size)  # Resize to match the model's input size
    image = image.astype("float") / 255.0  # Normalize the image
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model's input shape
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read the image from the request
    file = request.files["image"]
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Process the output
    pred_idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][pred_idx]
    label = "cancerous" if pred_idx == 1 else "non-cancerous"

    # Return the prediction as JSON
    return jsonify({"prediction": label, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
