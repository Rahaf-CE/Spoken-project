🎧 Audio Feature Extraction and Classification
This project performs audio classification using four different feature extraction techniques:

MFCC (Mel-Frequency Cepstral Coefficients)

ZCR (Zero Crossing Rate)

RMS Energy

Mel Spectrogram

A Random Forest Classifier is trained for each feature set to evaluate and compare performance.

🔍 Features
Loads audio file paths and labels from an Excel sheet

Extracts features using librosa

Trains and tests a classifier for each feature type

Provides classification reports and accuracy scores

Allows the user to upload and classify new audio files via a GUI

Visualization tools to compare model performance

Built with Tkinter GUI for easy interaction

🖼 GUI Preview
Upload and classify .wav files using MFCC, ZCR, RMS, or Mel Spectrogram features.

View accuracy and detailed classification report.

Compare performances using bar charts and summary text.

🧰 Technologies Used
Python

Librosa

Scikit-learn

Pandas

Tkinter

Matplotlib & Seaborn

📁 Dataset
Audio file paths and labels are loaded from an Excel file (data_set.xlsx). Ensure the paths are valid and point to .wav files.

