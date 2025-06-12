import streamlit as st
import os
import numpy as np
import librosa
import cv2
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from moviepy.editor

# Load models
# Correct way
cnn_model = load_model(r"C:\Users\Lenovo\OneDrive\Desktop\Projects\this_studio\InceptionV3Model.h5")
lstm_model = load_model(r"C:\Users\Lenovo\OneDrive\Desktop\Projects\this_studio\AudioLSTMModel (1).h5")

# Constants
FRAME_SIZE = (256, 256)
MAX_FRAMES = 30  # You can adjust based on performance
MFCC_FEATURES = 40
LSTM_INPUT_LENGTH = 130  # Depends on your training setup

def extract_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = frame / 255.0
        frames.append(frame)
        frame_count += 1

    cap.release()
    return np.array(frames)

def extract_audio_mfcc(video_path, max_len=LSTM_INPUT_LENGTH):
    clip = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES)
    mfcc = mfcc.T

    # Pad or trim to fixed length
    if mfcc.shape[0] > max_len:
        mfcc = mfcc[:max_len]
    else:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    os.remove(audio_path)
    return np.expand_dims(mfcc, axis=0)  # shape: (1, time_steps, features)

def predict_video_cnn(frames):
    preds = cnn_model.predict(frames)
    avg_pred = np.mean(preds)
    return avg_pred

def predict_audio_lstm(mfcc_features):
    pred = lstm_model.predict(mfcc_features)[0][0]
    return pred

def aggregate_predictions(video_score, audio_score):
    final_score = (video_score + audio_score) / 2
    return final_score

# Streamlit UI
st.title("Deepfake Video Detection")
st.write("Upload a video to analyze whether it's real or fake based on visual and audio clues.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Extracting frames and audio...")
    frames = extract_frames(tmp_path)
    mfcc = extract_audio_mfcc(tmp_path)

    st.success(f"Extracted {len(frames)} frames and audio MFCCs.")

    st.info("Running predictions...")

    video_score = predict_video_cnn(frames)
    audio_score = predict_audio_lstm(mfcc)

    final_score = aggregate_predictions(video_score, audio_score)

    real_percentage = (1 - final_score) * 100
    fake_percentage = final_score * 100

    st.markdown("### 🧠 Prediction Results")
    st.write(f"**Video-based prediction:** {video_score:.2f} (0=real, 1=fake)")
    st.write(f"**Audio-based prediction:** {audio_score:.2f} (0=real, 1=fake)")
    st.markdown(f"### Final Verdict: **{'Fake' if final_score > 0.5 else 'Real'}**")
    st.write(f"🟩 Real Confidence: {real_percentage:.2f}%")
    st.write(f"🟥 Fake Confidence: {fake_percentage:.2f}%")


