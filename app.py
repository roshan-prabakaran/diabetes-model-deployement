import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input
        features = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age]])

        # Scale
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        # Output
        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes ({probability:.2f} probability)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({probability:.2f} probability)")

    except Exception as e:
        st.error(f"Error: {e}")
