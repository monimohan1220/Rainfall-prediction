import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🌧️ Rainfall Prediction App")
st.write("Enter the weather conditions to predict whether it will rain tomorrow:")

# Input features
min_temp = st.number_input("Min Temperature (°C)", value=15.0)
max_temp = st.number_input("Max Temperature (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
humidity3pm = st.number_input("Humidity at 3 PM (%)", value=55.0)
pressure9am = st.number_input("Pressure at 9 AM (hPa)", value=1010.0)
cloud9am = st.number_input("Cloud at 9 AM (0-9)", value=4.0)
windspeed9am = st.number_input("Wind Speed at 9 AM (km/h)", value=15.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[min_temp, max_temp, rainfall, humidity3pm, pressure9am, cloud9am, windspeed9am]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction:")
    st.success("✅ Yes, it will rain tomorrow." if prediction == 1 else "❌ No, it won't rain tomorrow.")
    st.info(f"💧 Chance of rain: {probability * 100:.2f}%")
