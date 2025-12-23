import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.caption("👨‍💻 Developed by Shivang Mishra | Machine Learning Project")

st.write("Enter patient details to predict heart disease")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 50, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.number_input("Rest ECG (0–2)", 0, 2)
thalach = st.number_input("Max Heart Rate Achieved", 50, 220)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.5)
slope = st.number_input("Slope (0–2)", 0, 2)
ca = st.number_input("CA (0–4)", 0, 4)
thal = st.number_input("Thal (1–3)", 1, 3)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal'
    ])

    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
