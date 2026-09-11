import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("linear_model(2).joblib")

model = load_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("💻 Laptop Price Predictor")
st.write("Laptop price predictor model developed by MD.Nazmul Hasan Khan Mahmud")

st.divider()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8, step=1)
    rom_gb = st.number_input("ROM/Storage (GB)", min_value=32, max_value=4096, value=512, step=32)
    ssd_gb = st.number_input("SSD (GB)", min_value=0, max_value=4096, value=256, step=32)

with col2:
    processor_ghz = st.number_input("Processor Speed (GHz)", min_value=0.5, max_value=6.0, value=2.5, step=0.1)
    screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)

st.divider()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict Laptop price🔍", use_container_width=True):
    input_df = pd.DataFrame(
        [[ram_gb, rom_gb, ssd_gb, processor_ghz, screen_size]],
        columns=["RAM_GB", "ROM_GB", "SSD_GB", "Processor_GHz", "Screen_Size"],
    )

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated price in BDT : {prediction:,.2f}")
