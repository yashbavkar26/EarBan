# app.py — EarBan: AI Noise Monitor (SDG 11) with improved UI

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
from pydub import AudioSegment
from pydub.utils import which
import os

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
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-6) + 90  # approximate scaling
    return round(db, 2)

def classify_sound(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    top_class = class_names[scores.numpy().mean(axis=0).argmax()]
    return top_class

# ---------------- Streamlit UI ---------------- #
# Background image
with open("pexels-simon73-1323550.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

page_bg_img = f"""
<style>
body {{
background-image: url("data:image/jpg;base64,{encoded_string}");
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
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🎧 QuietCity – AI Noise Classifier")
st.markdown("""
Detect environmental sounds, measure decibel levels, and assess WHO risk levels.
""")

# ---------------- FFmpeg Check ---------------- #
ffmpeg_path = which("ffmpeg")
if ffmpeg_path is None:
    possible_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    if os.path.exists(possible_path):
        AudioSegment.converter = possible_path
    else:
        st.error(
            "FFmpeg not found! Please install it and add to PATH: https://ffmpeg.org/download.html"
        )
        st.stop()
else:
    AudioSegment.converter = ffmpeg_path

# ---------------- Audio Upload ---------------- #
uploaded_file = st.file_uploader("📂 Upload an audio file", type=["wav","mp3"])

if uploaded_file:
    try:
        # Convert any format to WAV
        audio = AudioSegment.from_file(uploaded_file)
        
        # Export to WAV in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Convert AudioSegment to NumPy array for analysis
        samples = np.array(audio.get_array_of_samples())

        # Convert to mono if stereo
        if audio.channels > 1:
         samples = samples.reshape((-1, audio.channels))
         samples = samples.mean(axis=1)

        # Normalize to float32 (-1.0 to 1.0)
        wav_data = samples.astype(np.float32) / (2**(8*audio.sample_width - 1))       
        
        # Analysis
        db_level = calculate_db(wav_data)
        sound_type = classify_sound(wav_data)
        
        # WHO risk
        if db_level < 55:
            status = "✅ Safe"
            color = "green"
        elif db_level < 70:
            status = "⚠️ Moderate"
            color = "orange"
        else:
            status = "🚨 Harmful"
            color = "red"

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔊 Sound Detected")
            st.markdown(f"**Class:** {sound_type}")
            st.markdown(f"**Decibel Level:** {db_level} dB")
        with col2:
            st.subheader("📊 WHO Risk Level")
            st.markdown(f"<span style='color:{color}; font-size:24px; font-weight:bold'>{status}</span>", unsafe_allow_html=True)

        # Plot waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wav_data, color="#1f77b4")
        ax.set_title(f"Waveform | {db_level} dB", color="white")
        ax.set_xlabel("Samples", color="white")
        ax.set_ylabel("Amplitude", color="white")
        ax.tick_params(colors='white')
        st.pyplot(fig)

        # CSV logging
        if st.button("💾 Save to CSV"):
            with open("noise_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([uploaded_file.name, sound_type, db_level, status])
            st.success("Data saved to noise_log.csv ✅")

    except Exception as e:
        st.error(f"Failed to process audio: {e}")
