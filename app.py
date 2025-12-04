from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import requests

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# ------------------------------
# Download model if missing
# ------------------------------

MODEL_PATH = "forest_model_imp.pkl"

# Your Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1D3dZRk8G0VLF9HADKXPMMq4c2OnKVi4Q"

# Auto-download model on Render if missing
if not os.path.exists(MODEL_PATH):
    print("⚠ Model not found — downloading from Google Drive...")
    try:
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    except Exception as e:
        print("❌ Failed to download model:", e)
        raise FileNotFoundError("Model could not be downloaded!")


# ------------------------------
# Load Model
# ------------------------------
model = joblib.load(MODEL_PATH)

# Model training feature order
model_features = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Wilderness_Area1', 'Wilderness_Area2',
    'Wilderness_Area3', 'Wilderness_Area4'
] + [f"Soil_Type{i}" for i in range(1, 41)]

# Label mapping
cover_type_mapping = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ------------------------------
# Home Route (Serve frontend)
# ------------------------------
@app.route("/")
def home():
    try:
        return render_template("fore.html")
    except:
        return "<h3>Render backend is running successfully.</h3>", 200


# ------------------------------
# Predict Route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Base features
    user_input = {
        'Elevation': data["Elevation"],
        'Aspect': data["Aspect"],
        'Slope': data["Slope"],
        'Horizontal_Distance_To_Hydrology': data["HD_Hydrology"],
        'Vertical_Distance_To_Hydrology': data["VD_Hydrology"],
        'Horizontal_Distance_To_Roadways': data["HD_Roadways"],
        'Hillshade_9am': data["Hillshade9"],
        'Hillshade_Noon': data["HillshadeNoon"],
        'Hillshade_3pm': data["Hillshade3"],

        # Wilderness encoding
        'Wilderness_Area1': 1 if data["Wilderness"] == 1 else 0,
        'Wilderness_Area2': 1 if data["Wilderness"] == 2 else 0,
        'Wilderness_Area3': 1 if data["Wilderness"] == 3 else 0,
        'Wilderness_Area4': 1 if data["Wilderness"] == 4 else 0,
    }

    # Soil types 1–40
    for i in range(1, 41):
        user_input[f"Soil_Type{i}"] = 1 if i in data["SoilTypes"] else 0

    # Prepare input DF
    input_df = pd.DataFrame([user_input])
    input_df = input_df[model_features]

    # Predict
    prediction = model.predict(input_df)[0]
    forest_name = cover_type_mapping[prediction]

    return jsonify({
        "predicted_class": int(prediction),
        "forest_name": forest_name
    })


# ------------------------------
# Render production entry
# ------------------------------
def handler(event, context):
    return app(event, context)


# ------------------------------
# Local Run
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
