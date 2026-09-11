import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("weather_model(2).joblib")

model = load_model()

st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="centered")

st.title("🌡️🌍🌥 Heat Index Prediction App")
st.write("Predict **Heat index** by filling those parameter")

st.divider()

# ------------------------------
# Input fields (model এর ৭টি feature অনুযায়ী)
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", value=60.0, step=0.1)
    pressure = st.number_input("Atmospheric Pressure (hPa)", value=1013.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.1)

with col2:
    wind_direction = st.number_input("Wind Direction (°)", value=180.0, step=1.0)
    dew_point = st.number_input("Dew Point (°C)", value=18.0, step=0.1)
    cloud_cover = st.number_input("Cloud Cover (%)", value=40.0, step=0.1)

st.divider()

# ------------------------------
# Prediction
# ------------------------------
if st.button("predict heat index", use_container_width=True):
    features = np.array([[temperature, humidity, pressure, wind_speed,
                           wind_direction, dew_point, cloud_cover]])

    # This model was trained on degree-2 polynomial features
    # (7 raw inputs -> 36 expanded terms), so we must expand
    # the raw inputs the same way before predicting.
    poly = PolynomialFeatures(degree=2, include_bias=True)
    features_poly = poly.fit_transform(features)

    prediction = model.predict(features_poly)[0]

    st.success(f"🌡️ Predicted Heat Index: **{prediction:.2f}°C**")

st.divider()
st.caption("Model: polynomial linear Regression | Target: Heat Index | Features: Temperature, Humidity, "
           "Atmospheric Pressure, Wind_Speed, Wind_Direction, Dew_Point, Cloud_Cover | this model is developed by MD.Nazmul Hasan Khan Mahmud")
