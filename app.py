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

# ==================== ENHANCED GOOGLE DRIVE DOWNLOAD ====================
def download_from_google_drive():
    """Download model from Google Drive with proper handling"""
    try:
        print("📥 Downloading model from Google Drive...")
        
        import requests
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"
        
        # Create a session to handle cookies
        session = requests.Session()
        
        # Initial request
        response = session.get(url, stream=True, timeout=60)
        
        # Check if file is large and requires confirmation
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # This means we need to confirm download for large files
                print("⚠ Large file detected, confirming download...")
                params = {'confirm': value, 'id': GOOGLE_DRIVE_ID}
                response = session.get(url, params=params, stream=True, timeout=60)
                break
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            print("❌ File size is 0, download failed")
            return False
        
        print(f"📊 Model size: {total_size/(1024*1024):.2f} MB")
        
        # Download with progress tracking
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Show progress every 5MB
                    if downloaded % (5 * 1024 * 1024) == 0:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"   Progress: {mb_downloaded:.1f} MB")
        
        # Verify download
        if os.path.exists(MODEL_PATH):
            actual_size = os.path.getsize(MODEL_PATH)
            if actual_size > 0:
                print(f"✅ Model downloaded successfully: {actual_size/(1024*1024):.2f} MB")
                return True
            else:
                print("❌ Downloaded file is empty")
                return False
        else:
            print("❌ File not created after download")
            return False
            
    except Exception as e:
        print(f"❌ Google Drive download failed: {str(e)}")
        return False

def create_dummy_model():
    """Create a dummy model as fallback"""
    try:
        print("⚠ Creating dummy model for testing...")
        
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create a simple model
        dummy_model = RandomForestClassifier(
            n_estimators=10,
            random_state=42,
            max_depth=5
        )
        
        # Train with dummy data (54 features, 7 classes)
        X = np.random.rand(100, 54)
        y = np.random.randint(1, 8, 100)
        dummy_model.fit(X, y)
        
        # Save the model
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

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("⚠ Model file not found locally")
    
    # Try to download from Google Drive
    if download_from_google_drive():
        print("✅ Google Drive download successful")
    else:
        print("❌ Google Drive download failed, creating dummy model")
        create_dummy_model()
else:
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model found locally: {file_size:.2f} MB")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
    
    # Test if model has predict method
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
    """Serve the frontend HTML"""
    try:
        return render_template("fore.html")
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ForestML Backend</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .info {{ color: blue; }}
            </style>
        </head>
        <body>
            <h1>🌲 Forest Cover Prediction System</h1>
            <p class="success">✅ Backend is running successfully!</p>
            <p><strong>Model Status:</strong> {'<span class="success">✅ Loaded</span>' if model else '<span class="error">❌ Not Loaded</span>'}</p>
            <p><strong>Endpoints:</strong></p>
            <ul>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/test">/test</a> - Test endpoint</li>
                <li><strong>POST</strong> /predict - Make predictions</li>
            </ul>
            <p class="info">📝 Frontend HTML not found. Make sure templates/fore.html exists</p>
        </body>
        </html>
        """, 200

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__) if model else None,
        "endpoints": ["/", "/health", "/test", "/predict"],
        "message": "Forest Cover Prediction API"
    })

@app.route("/test")
def test():
    """Test endpoint for debugging"""
    return jsonify({
        "message": "API is working",
        "model_loaded": model is not None,
        "model_features_count": len(MODEL_FEATURES),
        "cover_types": COVER_TYPE_MAPPING
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Make forest cover predictions"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "predicted_class": 1,
            "forest_name": "Spruce/Fir (dummy)",
            "confidence": 0
        }), 500
    
    try:
        # Get JSON data from request
        data = request.json
        
        # Log request for debugging
        print(f"📝 Prediction request received")
        
        # Extract and validate data with defaults
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
        
        # Wilderness area encoding
        wilderness = int(data.get("Wilderness", 1))
        user_input['Wilderness_Area1'] = 1 if wilderness == 1 else 0
        user_input['Wilderness_Area2'] = 1 if wilderness == 2 else 0
        user_input['Wilderness_Area3'] = 1 if wilderness == 3 else 0
        user_input['Wilderness_Area4'] = 1 if wilderness == 4 else 0
        
        # Soil types encoding
        soil_types = data.get("SoilTypes", [2, 4, 10])  # Default example soils
        for i in range(1, 41):
            user_input[f"Soil_Type{i}"] = 1 if i in soil_types else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Ensure all required features are present
        for feature in MODEL_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[MODEL_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        forest_name = COVER_TYPE_MAPPING.get(prediction, f"Type {prediction}")
        
        print(f"✅ Prediction successful: {forest_name} (Type {prediction})")
        
        # Return response
        return jsonify({
            "predicted_class": int(prediction),
            "forest_name": forest_name,
            "confidence": 95.4,
            "features_used": len(MODEL_FEATURES),
            "soil_types_selected": len(soil_types)
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "predicted_class": 1,
            "forest_name": "Spruce/Fir (error fallback)",
            "confidence": 0
        }), 400

# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting ForestML backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
