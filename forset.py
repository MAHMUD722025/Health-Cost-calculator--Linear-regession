"""
Weather Condition Predictor
Streamlit app for the Random Forest weather-condition model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Weather Condition Predictor",
    page_icon="🌤️",
    layout="centered",
)

# ---------------------------------------------------------
# Model
# IMPORTANT: the actual file in this repository is
# `forest_model(2).joblib`.
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "forest_model(2).joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()


model = load_model()

# Get the feature order stored by scikit-learn when available.
FEATURE_NAMES = list(
    getattr(
        model,
        "feature_names_in_",
        ["day_temp", "humidity", "cloud_cover", "uv_index"],
    )
)

CLASS_NAMES = list(getattr(model, "classes_", []))

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌤️ Weather Condition Predictor")
st.write(
    "Enter the weather readings below and the Random Forest model "
    "will predict the most likely condition."
)

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    day_temp = st.number_input(
        "Day Temperature (°C)",
        value=25.0,
        step=0.5,
        format="%.1f",
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
    )

with col2:
    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )
    uv_index = st.number_input(
        "UV Index",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):
    values = {
        "day_temp": day_temp,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "uv_index": uv_index,
    }

    # Keep exactly the same feature order used during training.
    try:
        X = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    except KeyError as e:
        st.error(
            f"The model expects an unsupported feature: {e}. "
            f"Model features: {FEATURE_NAMES}"
        )
        st.stop()

    try:
        prediction = model.predict(X)[0]

        st.success(f"🌤️ Predicted Weather Condition: **{prediction}**")

        # Show probabilities only when the model supports them.
        if hasattr(model, "predict_proba") and CLASS_NAMES:
            probabilities = model.predict_proba(X)[0]
            results = (
                pd.DataFrame(
                    {"Condition": CLASS_NAMES, "Probability": probabilities}
                )
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )

            st.subheader("Prediction probabilities")
            st.bar_chart(results.set_index("Condition")["Probability"])
            st.dataframe(
                results.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.expander("Model info"):
    st.write(f"**Features expected:** {FEATURE_NAMES}")
    if CLASS_NAMES:
        st.write(f"**Classes:** {CLASS_NAMES}")
    if hasattr(model, "n_estimators"):
        st.write(f"**Number of trees:** {model.n_estimators}")

st.caption(
    "Model developed by MD.Nazmul Hasan Khan Mahmud"
)
