# predict_from_audio.py

import numpy as np
import pandas as pd
import librosa
import pickle
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
AUDIO_FILE = 'ANGEL.wav'                     # Your test audio file
MODEL_PATH = 'genre_classifier.keras'  # Trained model path
ENCODER_PATH = 'label_encoder.pkl'     # Trained label encoder path
TRAINING_CSV = 'features_30_sec.csv'         # For extracting expected column order
SAMPLE_RATE = 22050
DURATION = 30
# -----------------------------------------

# ✅ Step 1: Load trained model
print("📥 Loading trained model...")
model = load_model(MODEL_PATH)

# ✅ Step 2: Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
print("✅ Label encoder loaded.")

# ✅ Step 3: Load expected column structure from training dataset
df_train = pd.read_csv(TRAINING_CSV)
expected_columns = df_train.drop(columns=['filename', 'label']).columns.tolist()
print(f"✅ Expected feature count: {len(expected_columns)}")

# ✅ Step 4: Load and preprocess audio file
print(f"🎧 Loading audio file: {AUDIO_FILE}")
signal, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE, duration=DURATION)

if len(signal) < SAMPLE_RATE * DURATION:
    padding = SAMPLE_RATE * DURATION - len(signal)
    signal = np.pad(signal, (0, padding), mode='constant')

# ✅ Step 5: Extract features from audio
def extract_features(signal, sr):
    features = {}

    features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))
    features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=signal, sr=sr))

    features['rms_mean'] = np.mean(librosa.feature.rms(y=signal))
    features['rms_var'] = np.var(librosa.feature.rms(y=signal))

    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=signal, sr=sr))

    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=signal, sr=sr))

    features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
    features['rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=signal, sr=sr))

    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y=signal))

    harmony = librosa.effects.harmonic(signal)
    percussive = librosa.effects.percussive(signal)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(percussive)
    features['perceptr_var'] = np.var(percussive)

    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
    features['tempo'] = tempo

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i - 1])
        features[f'mfcc{i}_var'] = np.var(mfcc[i - 1])

    return pd.DataFrame([features])

features = extract_features(signal, sr)

# ✅ Step 6: Add any missing columns & reorder
for col in expected_columns:
    if col not in features.columns:
        print(f"⚠️ Missing feature: {col} — Adding with 0.0")
        features[col] = 0.0

features = features[expected_columns]

# ✅ Final shape check
print("✅ Final features shape:", features.shape)

# ✅ Step 7: Predict
print("🔮 Predicting genre...")
pred_probs = model.predict(features)
pred_class_index = np.argmax(pred_probs, axis=1)[0]
pred_genre = label_encoder.inverse_transform([pred_class_index])[0]

# ✅ Output result
print(f"\n🎼 Predicted Genre: {pred_genre}")