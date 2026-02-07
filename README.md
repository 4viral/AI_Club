#Project by Aviral Srivastava 2025B2PS0925P


# 🎙️ Speech Emotion Recognition (SER) using 2D CNNs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Librosa](https://img.shields.io/badge/Librosa-0.10.x-green.svg)](https://librosa.org/)

This project implements a Speech Emotion Recognition (SER) system that treats audio signals as images. By converting raw audio into **Log-Mel Spectrograms**, we leverage a **2D Convolutional Neural Network (CNN)** to identify human sentiment from 3-second speech samples.

## 🎯 Project Objective
The goal is to classify audio from the **RAVDESS** dataset into 8 emotion categories:
* `Neutral`, `Calm`, `Happy`, `Sad`, `Angry`, `Fearful`, `Disgust`, `Surprised`.

## 🛠️ Technical Pipeline

### 1. Preprocessing & Feature Engineering
* **Audio Cleaning:** Used `librosa.effects.trim` to remove silence/dead air.
* **Feature Extraction:** Transformed waveforms into **Log-Mel Spectrograms** (128 Mel bands).
* **Uniform Shaping:** All spectrograms were padded or truncated to a fixed width of `300` time steps (approx. 3 seconds).
* **Normalization:** Applied **Z-score normalization** `(x - mean) / std` to ensure consistent spectral energy levels.
* **Data Augmentation:** To prevent overfitting on the small dataset, techniques like Noise Injection, Pitch Shifting, and Time Stretching were applied.

### 2. Model Architecture
Built a custom 2D CNN featuring:
* **Convolutional Blocks:** Multiple `Conv2D` layers with `BatchNormalization` for training stability.
* **Pooling:** `MaxPooling2D` for spatial downsampling.
* **Regularization:** `Dropout` layers and `GlobalAveragePooling2D` to prevent the model from memorizing specific actor voices.
* **Output:** A Softmax dense layer for 8-class emotion probability.

### 3. Evaluation Metrics
* **Macro F1-Score:** Used as the primary metric to account for class balance.
* **Confusion Matrix:** Analyzed to identify "Emotional Proximity" (e.g., confusion between Calm and Neutral).
* **Bias Analysis:** Conducted a performance audit comparing **Male vs. Female** speakers to detect pitch-based accuracy bias.

## 📂 Project Structure
```text
├── SER.ipynb               # Full EDA, Training, and Bias Analysis
├── emotion_model.h5        # Best performing model weights
├── predict.py              # Script for live inference on unseen .wav files
├── requirements.txt        # Dependencies (Librosa, TensorFlow, etc.)
└── README.md               # Project documentation
