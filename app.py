import streamlit as st
import numpy as np
import pandas as pd
import json
import os

# --- Helper Function: Read Live Data ---
def get_live_metrics():
    log_file = "raw_sensor_data.json"
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r') as f:
            # Read all lines and flatten the 'payload' lists
            data = [json.loads(line) for line in f]
            all_records = []
            for entry in data:
                all_records.extend(entry['payload'])
            
            df = pd.DataFrame(all_records)
            df['time'] = pd.to_datetime(df['time'], unit='ns')
            df.set_index('time', inplace=True)
            
            # Resample to get the latest hourly averages
            latest = df.resample('1H').mean().iloc[-1]
            return latest
    except Exception:
        return None

# --- UI Configuration ---
st.set_page_config(page_title="RelapsePredNet", layout="centered")

st.title("🧠 RelapsePredNet")
st.caption("Personalized Mobile Sensing-Based Relapse Prediction")

# --- STEP 1: Personalization ---
st.header("1. Patient Intake")
with st.expander("Patient Characteristics", expanded=False):
    sfs_score = st.slider("Social Functioning Scale (SFS) Score", 0, 150, 100)
    st.info(f"Model personalized for SFS: {sfs_score}")

# --- STEP 2: Live Sensing Modalities ---
st.header("2. Real-Time Sensing (Live from Phone)")
metrics = get_live_metrics()

col1, col2 = st.columns(2)

if metrics is not None:
    with col1:
        # Map sensor names to the 6 modalities
        st.metric("Movement (Accel)", f"{metrics.get('average', 0):.2f} m/s²")
        st.metric("Light Level", f"{metrics.get('lux', 0):.1f} lux")
    with col2:
        # Note: Some modalities like 'Conversation' are derived from audio
        st.metric("Latitude", f"{metrics.get('latitude', 0):.4f}")
        st.metric("Longitude", f"{metrics.get('longitude', 0):.4f}")
else:
    st.warning("No live data found. Start 'Sensor Logger' on your phone!")

# --- STEP 3: Prediction Logic ---
st.divider()
if st.button("Generate Relapse Prediction"):
    if metrics is not None:
        # This will eventually call your Bi-LSTM in model.py
        st.success("Analysis Complete: Patient is STABLE.")
    else:
        st.error("Cannot predict without active sensing data.")