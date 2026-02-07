import os
import sys
import librosa
import numpy as np
import tensorflow as tf

SR = 22050
N_MELS = 128
MAX_LEN = 300
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def preprocess_input(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=20)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel[np.newaxis, ..., np.newaxis]
    return mel

def run_inference(model_path, audio_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    input_data = preprocess_input(audio_path)
    
    predictions = model.predict(input_data, verbose=0)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]
    
    print("-" * 30)
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Predicted Emotion: {EMOTIONS[idx].upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <audio_path>")
    else:
        run_inference(sys.argv[1], sys.argv[2])