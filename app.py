# app.py — EarBan: AI Noise Monitor (SDG 11) with improved UI + Heatmap

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
import csv
import requests
import base64
import os
import librosa
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from guidelines import get_who_guidelines, get_emergency_contacts

# ---------------- Load Model + Class Names ---------------- #
@st.cache_resource
def load_model():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

@st.cache_resource
def load_class_names():
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    response = requests.get(url)
    class_names = []
    if response.status_code == 200:
        reader = csv.reader(response.text.splitlines())
        next(reader)  # skip header
        class_names = [row[2] for row in reader]
    return class_names

yamnet_model = load_model()
class_names = load_class_names()

# ---------------- Helper Functions ---------------- #
def calculate_db(audio):
    """Convert waveform to approximate dB SPL scale"""
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-6)  # dBFS
    db_spl = db + 94  # calibration offset (approximate for SPL)
    return round(db_spl, 2)

def classify_sound(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    scores_np = scores.numpy()
    top_class_idx = np.argmax(np.mean(scores_np, axis=0))
    return class_names[top_class_idx]

# ---------------- Streamlit UI ---------------- #
# Background image
with open("pexels-simon73-1323550.jpg", "rb") as image_file:
    encoded_bg = base64.b64encode(image_file.read()).decode()

# Logo image
with open("logo.jpg", "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()

page_style = f"""
<style>
body {{
background-image: url("data:image/jpg;base64,{encoded_bg}");
background-size: cover;
background-attachment: fixed;
}}
.stButton>button {{
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}}
.stApp {{
    color: #FFFFFF;
    background-color: rgba(0,0,0,0.5);
    padding: 2rem;
    border-radius: 15px;
}}
/* Logo positioning */
.logo-container {{
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 9999;
}}
.logo-container img {{
    width: 80px;
    border-radius: 10px;
    box-shadow: 0px 0px 6px rgba(0,0,0,0.5);
}}
</style>

<div class="logo-container">
    <img src="data:image/jpg;base64,{encoded_logo}">
</div>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.title("  QuietCity – AI Noise Classifier")
st.markdown("""
Detect environmental sounds, measure decibel levels, and assess WHO risk levels.
""")
# WHO Guidelines and Emergency Contacts
st.sidebar.title("Noise Pollution Help")

# WHO Guidelines Dropdown
with st.sidebar.expander("📘 WHO Guidelines"):
    guidelines = get_who_guidelines()
    for k, v in guidelines.items():
        st.markdown(f"**{k}:** {v}")

# Emergency Contacts Dropdown
with st.sidebar.expander("☎️ Emergency Contacts"):
    contacts = get_emergency_contacts()
    for k, v in contacts.items():
        st.markdown(f"**{k}:** {v}")


# File upload
uploaded_file = st.file_uploader("📂 Upload a WAV file", type=["wav"])

if uploaded_file:
    wav_bytes = io.BytesIO(uploaded_file.read())
    sr, wav_data = wavfile.read(wav_bytes)

    # Convert to mono
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=1)

    # Normalize to float32
    wav_data = wav_data.astype(np.float32)
    if np.max(np.abs(wav_data)) > 0:
        wav_data = wav_data / np.max(np.abs(wav_data))

    # Resample to 16kHz for YAMNet
    wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
    sr = 16000

    # Run analysis
    db_level = calculate_db(wav_data)
    sound_type = classify_sound(wav_data)

    # WHO threshold classification
    if db_level < 55:
        status = "✅ Safe"
        color = "green"
    elif db_level < 70:
        status = "⚠️ Moderate"
        color = "orange"
    else:
        status = "🚨 Harmful"
        color = "red"

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔊 Sound Detected")
        st.markdown(f"**Class:** {sound_type}")
        st.markdown(f"**Decibel Level:** {db_level} dB SPL")
    with col2:
        st.subheader("📊 WHO Risk Level")
        st.markdown(
            f"<span style='color:{color}; font-size:24px; font-weight:bold'>{status}</span>",
            unsafe_allow_html=True
        )

    # Plot waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wav_data, color="#1f77b4")
    ax.set_title(f"Waveform | {db_level} dB SPL", color="white")
    ax.set_xlabel("Samples", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Location input
    lat = st.number_input("📍 Enter Latitude", value=19.0760, format="%.6f")
    lon = st.number_input("📍 Enter Longitude", value=72.8777, format="%.6f")

    # CSV logging
    if st.button("💾 Save to CSV"):
        log_file = "noise_log.csv"
        file_exists = os.path.isfile(log_file)

        with open(log_file, "a", newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Filename", "Class", "Decibel Level", "Risk Status", "Latitude", "Longitude"])
            writer.writerow([uploaded_file.name, sound_type, db_level, status, lat, lon])
        st.success("Data saved to noise_log.csv ✅")

# ---------------- Heatmap Section ---------------- #
st.subheader("🌍 Noise Pollution Heatmap")

try:
    df = pd.read_csv("noise_log.csv")

    if not df.empty:
        # Center map at average coordinates
        map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        # Add heatmap based on decibel level
        heat_data = [[row["Latitude"], row["Longitude"], row["Decibel Level"]] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)

        st_folium(m, width=700, height=500)
    else:
        st.info("No data logged yet. Upload and save some noise samples.")
except FileNotFoundError:
    st.info("No data file found. Save some noise data first.")

# ---------------- Footer / Credits ---------------- #
st.markdown("""<hr style="margin-top:40px; margin-bottom:20px; border:1px solid #444;">""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; color: #ccc; font-size: 16px;">
        <strong>Created By:</strong> Team QuietStorm<br>
        <strong>Members:</strong> Vighnesh Patil &nbsp; | &nbsp; Yash Bavkar
    </div>
    """,
    unsafe_allow_html=True
)
