import streamlit as st
import numpy as np
import time

# --- Page Config ---
st.set_page_config(page_title="RelapsePredNet", layout="centered")

# --- Title Section ---
st.title("🧠 RelapsePredNet")
st.caption("Personalized Mobile Sensing-Based Relapse Prediction")
st.divider()

# --- STEP 1: Personalization Metric ---
st.header("1. Patient Intake")
with st.expander("Enter Patient Characteristics", expanded=True):
    # SFS is the best metric for personalization according to the paper
    sfs_score = st.slider("Social Functioning Scale (SFS) Score", 0, 150, 100)
    age = st.number_input("Patient Age", 18, 100, 35)
    st.info("The model will now select a training subset of 'similar' patients.")

# --- STEP 2: Sensing Modalities ---
st.header("2. Real-Time Sensing Data (Hourly Averages)")
col1, col2 = st.columns(2)

with col1:
    st.write("**Social & Environment**")
    conv_dur = st.metric("Conversation Duration", "15 min", "-12%") #
    light_lvl = st.metric("Light Exposure", "350 lux", "+5%") #
    audio_vol = st.metric("Ambient Volume", "45 dB", "Stable") #

with col2:
    st.write("**Mobility & Activity**")
    dist_trav = st.metric("Distance Traveled", "0.8 km", "-45%") #
    accel_mag = st.metric("Movement (Accel)", "1.2 m/s²", "-10%") #
    screen_use = st.metric("Screen Usage", "5.2 hrs", "+15%") #

# --- STEP 3: Prediction Logic ---
st.divider()
if st.button("Generate Relapse Prediction"):
    with st.status("Running Bi-LSTM Model on 30-day window...", expanded=True) as status:
        time.sleep(1)
        st.write("Fetching personalized weights based on SFS...")
        time.sleep(1)
        st.write("Analyzing behavioral anomalies...")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Simulated output based on the 108% anomaly finding
    risk_score = 0.72 
    st.subheader(f"Calculated Relapse Risk: {risk_score*100:.1f}%")
    
    if risk_score > 0.5:
        st.error("⚠️ HIGH RISK: Behavioral patterns match historical relapse indicators.")
        st.write("- **Primary Driver**: Significant drop in 'Conversation' and 'Distance'.")
    else:
        st.success("✅ STABLE: Current behaviors are within normal range for this patient.")

st.divider()
st.caption("Developed for CSE Final Year Project | Based on Lamichhane et al. (2023)")