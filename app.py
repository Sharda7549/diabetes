import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load your trained diabetes model safely
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "diabetes.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Custom CSS with gradient background
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
    font-family: 'Arial', sans-serif;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.css-1d391kg {background: transparent;}
.stButton button {
    background: #ffffff;
    color: #2a5298;
    border-radius: 12px;
    padding: 0.6em 1.5em;
    font-size: 16px;
    font-weight: bold;
    border: none;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    transition: 0.3s;
}
.stButton button:hover {
    background: #2a5298;
    color: white;
}
input, .stNumberInput input {
    border-radius: 10px !important;
    padding: 8px !important;
    border: none !important;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Title & Description
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter patient details below to predict the likelihood of diabetes.</p>", unsafe_allow_html=True)

# -------------------------------
# Input Form
# -------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Glucose_BMI_Interaction = st.number_input("Glucose × BMI Interaction", min_value=0.0, format="%.2f")
        Glucose = st.number_input("Glucose", min_value=0.0, format="%.2f")
        BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
        BMI_DPF_Interaction = st.number_input("BMI × DiabetesPedigreeFunction", min_value=0.0, format="%.2f")

    with col2:
        Age = st.number_input("Age", min_value=0, step=1)
        Pregnancies_Age_Interaction = st.number_input("Pregnancies × Age Interaction", min_value=0.0, format="%.2f")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        BloodPressure = st.number_input("Blood Pressure", min_value=0.0, format="%.2f")

    submit = st.form_submit_button("🔍 Predict")

# -------------------------------
# Prediction
# -------------------------------
if submit:
    # Prepare feature vector
    features = np.array([[Glucose_BMI_Interaction, Glucose, BMI,
                          BMI_DPF_Interaction, Age, Pregnancies_Age_Interaction,
                          DiabetesPedigreeFunction, BloodPressure]])
    
    # Prediction
    prediction = model.predict(features)[0]

    # Check probability if available
    proba = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

    # Show result
    if prediction == 1:
        st.markdown("<h3 style='text-align:center; color:#ff4d4d;'>⚠️ High Risk: Likely Diabetic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center; color:#00ff99;'>✅ Low Risk: Unlikely Diabetic</h3>", unsafe_allow_html=True)

    # Show probability as progress bar
    if proba is not None:
        st.progress(int(proba*100))
        st.markdown(f"<p style='text-align:center; font-size:18px;'>Prediction confidence: <b>{proba*100:.2f}%</b></p>", unsafe_allow_html=True)
