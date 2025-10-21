import streamlit as st
import pandas as pd
import joblib
import socket

st.set_page_config(page_title="Health Insight AI", layout="centered")
st.title("Health Insight AI: Heart Disease Prediction")

# Load model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.write("Enter patient details below:")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)", [3, 6, 7])

if st.button("Predict"):
    df = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                      columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")

st.markdown("---")
st.caption("A predictive healthcare project by powered by MLOps workflow.")

# Button to open monitoring dashboard
# Get current hostname
hostname = socket.gethostname()

# Detect local environment
is_local = hostname in ["localhost", "127.0.0.1", "TU-63VC284"] or "local" in hostname.lower()

# Define URLs
local_link = "http://localhost:8502"
cloud_link = "https://health-insight-ai-monitoring-yourname.streamlit.app" 

# Select URL based on environment
dashboard_url = local_link if is_local else cloud_link

# Render button
st.markdown(
    f"""
    <a href="{dashboard_url}" target="_blank">
        <button style="background-color:#0078ff;color:white;padding:10px 20px;border:none;border-radius:6px;cursor:pointer;">
            Open Monitoring Dashboard
        </button>
    </a>
    """,
    unsafe_allow_html=True
)