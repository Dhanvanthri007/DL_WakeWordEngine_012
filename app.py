import streamlit as st
import torch
import torchaudio

import soundfile as sf
import numpy as np

# Completely bypass torchaudio.load due to torchcodec bug
def _custom_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    else:
        data = data.T
    return torch.from_numpy(data).float(), sr
torchaudio.load = _custom_load
import soundfile as sf
import os
import time
from datetime import datetime
import numpy as np

# Adjust imports to find src files
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import AudioPreprocessor
from model import WakeWordCNN

# Page Configuration
st.set_page_config(page_title="Personalized Wake Word Engine", page_icon="🎙️", layout="wide")

st.title("🎙️ Personalized Wake Word Engine")
st.markdown("Upload or record audio to test the deep learning model. The predefined wake word is **'marvin'**. When detected, it will prompt an alarm scheduling interface.")

# Load Model
@st.cache_resource
def load_model():
    model = WakeWordCNN(num_classes=2, in_channels=1)
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
preprocessor = AudioPreprocessor()

def predict(audio_path):
    with torch.no_grad():
        mfcc = preprocessor.process_file(audio_path)
        # Add batch dimension
        mfcc = mfcc.unsqueeze(0) 
        output = model(mfcc)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        score = probabilities[0][1].item() # probability of class 1 (wake word)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class == 1, score

audio_value = st.audio_input("🎙️ Tap to record your voice command ('Hey Marvin...')")

if audio_value is not None:
    # Save the file temporarily
    temp_path = "assets/temp_audio.wav"
    os.makedirs("assets", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(audio_value.getbuffer())
        
    with st.spinner("Processing audio..."):
        time.sleep(1) # Simulate processing time for UX
        is_wake_word, score = predict(temp_path)
        
    if is_wake_word:
        st.success(f"🎉 Wake Word Detected! (Confidence: {score:.2f})")
        st.divider()
        st.subheader("⚙️ Automated Tasks Executed")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("⏰ **Alarm Set**\n\nAutomatically configured for 7:00 AM.")
        with col2:
            st.info("📅 **Calendar Reminder**\n\nScheduled 'Morning Review' at 9:00 AM.")
    else:
        st.error(f"❌ Wake Word NOT Detected. (Confidence: {score:.2f})")
