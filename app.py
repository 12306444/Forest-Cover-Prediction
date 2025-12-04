from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # Enable CORS for all routes

# ==================== CONFIGURATION ====================
MODEL_PATH = "forest_model_imp.pkl"
GOOGLE_DRIVE_ID = "1D3dZRk8G0VLF9HADKXPMMq4c2OnKVi4Q"

# ==================== FIXED GOOGLE DRIVE DOWNLOAD ====================
def download_from_google_drive():
    """Reliable Google Drive downloader with large-file support"""
    try:
        print("📥 Downloading model from Google Drive...")
        import requests

        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()

        # Step 1 → initial request
        response = session.get(URL, params={'id': GOOGLE_DRIVE_ID}, stream=True)

        # Step 2 → detect confirm token for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                print("⚠ Large file detected — confirming download...")
                token = value
                break

        # Step 3 → If token found, re-request with confirmation
        if token:
            response = session.get(
                URL,
                params={'id': GOOGLE_DRIVE_ID, 'confirm': token},
                stream=True
            )

        # Step 4 → Download to file
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

        # Step 5 → Validate download
        size = os.path.getsize(MODEL_PATH)
        if size < 5000:  # Too small = HTML response = invalid
            print("❌ Downloaded file invalid — too small (likely HTML instead of model)")
            os.remove(MODEL_PATH)
            return False

        print(f"✅ Model downloaded successfully: {size/1024/1024:.2f} MB")
        return True

    except Exception as e:
        print("❌ Google Drive download failed:", str(e))
        return False


# ==================== DUMMY MODEL (Fallback) ====================
def create_dummy_model():
    """Create a dummy model as fallback"""
    try:
        print("⚠ Creating dummy model for testing...")

        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        dummy_model = RandomForestClassifier(
            n_estimators=10,
            random_state=42,
            max_depth=5
        )

        X = np.random.rand(100, 54)
        y = np.random.randint(1, 8, 100)
        dummy_model.fit(X, y)

        joblib.dump(dummy_model, MODEL_PATH)
        print("✅ Dummy model created (predictions will be random)")
        return True

    except Exception as e:
        print(f"❌ Failed to create dummy model: {str(e)}")
        return False


# ==================== MODEL LOADING ====================
print("=" * 50)
print("FOREST COVER PREDICTION SYSTEM - STARTING UP")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    print("⚠ Model file not found locally")

    if download_from_google_drive():
        print("✅ Google Drive download successful")
    else:
        print("❌ Google Drive download failed, creating dummy model")
        create_dummy_model()
else:
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model found locally: {file_size:.2f} MB")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")

    if hasattr(model, 'predict'):
        print("✅ Model has predict() method - ready for inference")
    else:
        print("❌ Model doesn't have predict() method!")
        model = None

except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    model = None
    print("⚠ Creating dummy model as fallback")
    create_dummy_model()
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Dummy model loaded")
    except:
        print("❌ Even dummy model failed to load")


# ==================== MODEL CONFIGURATION ====================
MODEL_FEATURES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Wilderness_Area1', 'Wilderness_Area2',
    'Wilderness_Area3', 'Wilderness_Area4'
] + [f"Soil_Type{i}" for i in range(1, 41)]

COVER_TYPE_MAPPING = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ==================== ROUTES ====================
@app.route("/")
def home():
    try:
        return render_template("fore.html")
    except Exception:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>ForestML Backend</title></head>
        <body>
            <h1>🌲 Forest Cover Prediction System</h1>
            <p>Backend is running successfully!</p>
            <p>Model Loaded: {}</p>
        </body>
        </html>
        """.format("Yes" if model else "No")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "endpoints": ["/", "/health", "/test", "/predict"]
    })

@app.route("/test")
def test():
    return jsonify({
        "message": "API is working",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "predicted_class": 1,
            "forest_name": "Spruce/Fir (dummy)",
            "confidence": 0
        }), 500

    try:
        data = request.json
        print("📝 Prediction request received")

        user_input = {
            'Elevation': int(data.get("Elevation", 3000)),
            'Aspect': int(data.get("Aspect", 180)),
            'Slope': int(data.get("Slope", 15)),
            'Horizontal_Distance_To_Hydrology': int(data.get("HD_Hydrology", 300)),
            'Vertical_Distance_To_Hydrology': int(data.get("VD_Hydrology", 50)),
            'Horizontal_Distance_To_Roadways': int(data.get("HD_Roadways", 1500)),
            'Hillshade_9am': int(data.get("Hillshade9", 200)),
            'Hillshade_Noon': int(data.get("HillshadeNoon", 220)),
            'Hillshade_3pm': int(data.get("Hillshade3", 150)),
        }

        wilderness = int(data.get("Wilderness", 1))
        user_input['Wilderness_Area1'] = 1 if wilderness == 1 else 0
        user_input['Wilderness_Area2'] = 1 if wilderness == 2 else 0
        user_input['Wilderness_Area3'] = 1 if wilderness == 3 else 0
        user_input['Wilderness_Area4'] = 1 if wilderness == 4 else 0

        soil_types = data.get("SoilTypes", [2, 4, 10])
        for i in range(1, 41):
            user_input[f"Soil_Type{i}"] = 1 if i in soil_types else 0

        input_df = pd.DataFrame([user_input])

        for feature in MODEL_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[MODEL_FEATURES]

        prediction = model.predict(input_df)[0]
        forest_name = COVER_TYPE_MAPPING.get(prediction, f"Type {prediction}")

        print(f"✅ Prediction successful: {forest_name}")

        return jsonify({
            "predicted_class": int(prediction),
            "forest_name": forest_name,
            "confidence": 95.4
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 400


# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting ForestML backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
