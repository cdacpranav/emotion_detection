import os
import wave
import datetime
import threading
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import pyaudio

# Load the trained emotion detection model
model = tf.keras.models.load_model("speech_emotion_model.h5", compile=False)  # Avoids auto-compiling
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
recording = False
frames = []


# Function to extract features from audio
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])


# Function to start recording
def start_recording():
    global recording, frames
    recording = True
    frames = []

    def record():
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()

    thread = threading.Thread(target=record)
    thread.start()


# Function to stop recording and save file
def stop_recording():
    global recording
    recording = False

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = f"Recording_{timestamp}.wav"

    # Save the recorded audio
    wave_file = wave.open(file_path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    st.success(f"Recording saved as {file_path}")

    return file_path  # Return saved file path


# Streamlit UI
st.title("Speech Emotion Detection")

st.write("🎤 Click **Start Recording** to begin speaking, and **Stop Recording** to save.")

# Create Streamlit buttons
if st.button("Start Recording 🎙️"):
    start_recording()
    st.write("Recording... Speak now!")

if st.button("Stop Recording ⏹️"):
    file_path = stop_recording()

    # Extract features and make prediction
    features = extract_audio_features(file_path)
    features = np.expand_dims(features, axis=(0, 2))

    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)

    # Emotion mapping
    emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
    st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")

# File Upload Option
st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])


if uploaded_file:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    features = extract_audio_features(file_path)
    features = np.expand_dims(features, axis=(0, 2))

    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)

    emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
    st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")