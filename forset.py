"""
Weather Condition Predictor (Streamlit)
========================================
Serves predictions from `forest_model_2_.joblib`, a scikit-learn
RandomForestClassifier trained to predict weather condition
(Cloudy / Cold / Hot / Rainy / Sunny) from 4 numeric features:
    - day_temp     (temperature, e.g. degrees C)
    - humidity     (%)
    - cloud_cover  (%)
    - uv_index     (0-11+)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "forest_model_2_.joblib")

st.set_page_config(page_title="Weather Condition Predictor", page_icon="🌍🌤️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

FEATURE_NAMES = list(getattr(model, "feature_names_in_",
                              ["day_temp", "humidity", "cloud_cover", "uv_index"]))
CLASS_NAMES = list(model.classes_)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🌤️ Weather Condition Predictor model developed by MD.Nazmul Hasan Khan Mahmud")
st.write(
    "Enter the day's readings below and the model will predict the most likely "
    "weather condition."
)

col1, col2 = st.columns(2)

with col1:
    day_temp = st.number_input("Day Temperature (°C)", value=25.0, step=0.5, format="%.1f")
    humidity = st.number_input("Humidity (%)", value=50.0, min_value=0.0, max_value=100.0, step=1.0)

with col2:
    cloud_cover = st.number_input("Cloud Cover (%)", value=20.0, min_value=0.0, max_value=100.0, step=1.0)
    uv_index = st.number_input("UV Index", value=5.0, min_value=0.0, max_value=15.0, step=0.5)

predict_clicked = st.button("Predict", type="primary")

st.divider()

if predict_clicked:
    values = {
        "day_temp": day_temp,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "uv_index": uv_index,
    }
    X = np.array([[values[name] for name in FEATURE_NAMES]])

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    results = (
        pd.DataFrame({"Condition": CLASS_NAMES, "Probability": proba})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader(f"Prediction: {prediction}")
    st.bar_chart(results.set_index("Condition")["Probability"])

    st.dataframe(
        results.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Fill in the values above and click **Predict** to see a result.")

with st.expander("Model info"):
    st.write(f"**Features expected:** {FEATURE_NAMES}")
    st.write(f"**Classes:** {CLASS_NAMES}")
    st.write(f"**Number of trees:** {model.n_estimators}")
