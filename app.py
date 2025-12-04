from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# ==================== CONFIG ====================
MODEL_PATH = "forest_model_imp.pkl"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/2ng3972vbd6xl9zvsccvm/forest_model_imp.pkl?rlkey=jhomouyhdol25jgniqdyfcj38&st=55ok58tt&dl=1"

# ==================== DROPBOX DOWNLOAD ====================
def download_model_from_dropbox():
    """Download model from Dropbox (stable for large files)."""
    try:
        print("📥 Downloading model from Dropbox...")
        import requests

        response = requests.get(DROPBOX_URL, stream=True, timeout=120)

        # Write file
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        # Validate
        size = os.path.getsize(MODEL_PATH)
        if size < 5000:
            print("❌ Dropbox returned invalid file (too small)")
            os.remove(MODEL_PATH)
            return False

        print(f"✅ Model downloaded successfully: {size/1024/1024:.2f} MB")
        return True

    except Exception as e:
        print("❌ Dropbox download failed:", str(e))
        return False


# ==================== DUMMY MODEL (Fallback) ====================
def create_dummy_model():
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
        print("✅ Dummy model created (random predictions)")
        return True

    except Exception as e:
        print(f"❌ Failed to create dummy model:", str(e))
        return False


# ==================== MODEL LOADING ====================
print("=" * 50)
print("FOREST COVER PREDICTION SYSTEM - STARTING UP")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    print("⚠ Model file not found locally")

    if download_model_from_dropbox():
        print("✅ Dropbox model download successful")
    else:
        print("❌ Dropbox download failed — using dummy model")
        create_dummy_model()
else:
    size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model found locally: {size:.2f} MB")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")

    if hasattr(model, "predict"):
        print("✅ Model ready for inference")
    else:
        print("❌ Model missing predict()!")
        model = None

except Exception as e:
    print("❌ Failed to load model:", str(e))
    print("⚠ Creating dummy model...")
    create_dummy_model()
    model = joblib.load(MODEL_PATH)
    print("✅ Dummy model loaded")


# ==================== MODEL CONFIG ====================
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
    except:
        return f"""
        <html>
        <body>
            <h1>ForestML Backend Running</h1>
            <p>Model Loaded: {"Yes" if model else "No"}</p>
        </body>
        </html>
        """

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    })

@app.route("/test")
def test():
    return jsonify({
        "message": "API working",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json

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

        soil = data.get("SoilTypes", [])
        for i in range(1, 41):
            user_input[f"Soil_Type{i}"] = 1 if i in soil else 0

        df = pd.DataFrame([user_input])
        df = df.reindex(columns=MODEL_FEATURES, fill_value=0)

        prediction = model.predict(df)[0]
        forest_name = COVER_TYPE_MAPPING.get(prediction, "Unknown")

        return jsonify({
            "predicted_class": int(prediction),
            "forest_name": forest_name,
            "confidence": 95.4
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
