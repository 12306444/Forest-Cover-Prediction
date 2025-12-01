from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # optional, but safe

# Serve fore.html (your frontend)
@app.route("/")
def home():
    return render_template("fore.html")


# ------------------------------
# Load the retrained model
# ------------------------------
model = joblib.load("forest_model_imp.pkl")

# Feature list used during retraining
model_features = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Wilderness_Area1', 'Wilderness_Area2',
    'Wilderness_Area3', 'Wilderness_Area4'
] + [f"Soil_Type{i}" for i in range(1, 41)]

# Forest type mapping
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
# Predict Route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Step 1: Base features
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

        # Wilderness one-hot encoding
        'Wilderness_Area1': 1 if data["Wilderness"] == 1 else 0,
        'Wilderness_Area2': 1 if data["Wilderness"] == 2 else 0,
        'Wilderness_Area3': 1 if data["Wilderness"] == 3 else 0,
        'Wilderness_Area4': 1 if data["Wilderness"] == 4 else 0,
    }

    # Step 2: Soil types (1–40)
    for i in range(1, 41):
        user_input[f"Soil_Type{i}"] = 1 if i in data["SoilTypes"] else 0

    # Step 3: Convert to dataframe in correct order
    input_df = pd.DataFrame([user_input])
    input_df = input_df[model_features]

    # Step 4: Predict
    prediction = model.predict(input_df)[0]
    forest_name = cover_type_mapping[prediction]

    return jsonify({
        "predicted_class": int(prediction),
        "forest_name": forest_name
    })


# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
