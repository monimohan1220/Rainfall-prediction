import streamlit as st
import pickle
import numpy as np
import sklearn as sk

# Load the trained model
with open('rainfall_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.title("🌧️ Rainfall Prediction App")

st.markdown("""
Enter the weather parameters to predict rainfall.
""")

# Input features
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)

# You can add more inputs if your model uses more features

# Predict button
if st.button("Predict Rainfall"):
    features = np.array([[temperature, humidity, pressure, wind_speed]])
    prediction = model.predict(features)[0]
    st.success(f"🌦️ Predicted Rainfall: {prediction:.2f} mm")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Machine Learning")
