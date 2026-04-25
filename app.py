from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/")
def home():
    return "Diabetes Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features (adjust order based on your dataset)
        features = [
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]

        # Convert to numpy array
        features = np.array([features])

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        # Optional probability
        probability = model.predict_proba(scaled_features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
