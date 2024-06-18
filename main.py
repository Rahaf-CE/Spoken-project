import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import time
from tkinter import scrolledtext, filedialog, Label, Toplevel
import tkinter as tk
from tkinter import ttk
import seaborn as sns
import matplotlib.pyplot as plt


# Function to extract MFCC features from an audio file
def extract_mfcc_features(file_name):
    try:
        start_time = time.time()
        audio, sample_rate = librosa.load(file_name, sr=None)  # Load with the original sampling rate
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        elapsed_time = time.time() - start_time
        print(f"Processed {file_name} in {elapsed_time:.2f} seconds")
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None


# Function to extract ZCR features from an audio file
def extract_zcr_features(file_name):
    try:
        start_time = time.time()
        audio, sample_rate = librosa.load(file_name, sr=None)  # Load with the original sampling rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)
        elapsed_time = time.time() - start_time
        print(f"Processed {file_name} in {elapsed_time:.2f} seconds")
        return zcr_mean
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Function to extract RMS Energy features from an audio file
def extract_rms_features(file_name):
    try:
        start_time = time.time()
        audio, sample_rate = librosa.load(file_name, sr=None)  # Load with the original sampling rate
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        elapsed_time = time.time() - start_time
        print(f"Processed {file_name} in {elapsed_time:.2f} seconds")
        return rms_mean
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Function to extract Mel Spectrogram features from an audio file
def extract_mel_spectrogram_features(file_name):
    try:
        start_time = time.time()
        audio, sample_rate = librosa.load(file_name, sr=None)  # Load with the original sampling rate
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        elapsed_time = time.time() - start_time
        print(f"Processed {file_name} in {elapsed_time:.2f} seconds")
        return np.mean(mel_spectrogram_db.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Read dataset from Excel file
dataset = pd.read_excel("C:\\Users\\User\\Desktop\\Data_set\\data_set.xlsx")  # Excel file path

# Extract labels and file paths
labels = dataset.iloc[:, 0].values
audio_files = dataset.iloc[:, 1].values

# Extract MFCC features and prepare the dataset
features_mfcc = []
valid_labels_mfcc = []

for file, label in zip(audio_files, labels):
    if os.path.isfile(file):
        feature = extract_mfcc_features(file)
        if feature is not None:
            features_mfcc.append(feature)
            valid_labels_mfcc.append(label)
    else:
        print(f"File not found: {file}")

X_mfcc = np.array(features_mfcc)
y_mfcc = np.array(valid_labels_mfcc)

# Split the dataset into training and testing sets for MFCC
X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(X_mfcc, y_mfcc, test_size=0.33, random_state=42)

# Initialize and train the classifier for MFCC
classifier_mfcc = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_mfcc.fit(X_train_mfcc, y_train_mfcc)

# Make predictions on the test set for MFCC
y_pred_mfcc = classifier_mfcc.predict(X_test_mfcc)

# Evaluate the classifier for MFCC
accuracy_mfcc = accuracy_score(y_test_mfcc, y_pred_mfcc)
classification_rep_mfcc = classification_report(y_test_mfcc, y_pred_mfcc, zero_division=0)


# Extract ZCR features and prepare the dataset
features_zcr = []
valid_labels_zcr = []

for file, label in zip(audio_files, labels):
    if os.path.isfile(file):
        feature = extract_zcr_features(file)
        if feature is not None:
            features_zcr.append(feature)
            valid_labels_zcr.append(label)
    else:
        print(f"File not found: {file}")

X_zcr = np.array(features_zcr).reshape(-1, 1)
y_zcr = np.array(valid_labels_zcr)

# Split the dataset into training and testing sets for ZCR
X_train_zcr, X_test_zcr, y_train_zcr, y_test_zcr = train_test_split(X_zcr, y_zcr, test_size=0.44, random_state=42)

# Initialize and train the classifier for ZCR
classifier_zcr = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_zcr.fit(X_train_zcr, y_train_zcr)

# Make predictions on the test set for ZCR
y_pred_zcr = classifier_zcr.predict(X_test_zcr)

# Evaluate the classifier for ZCR
accuracy_zcr = accuracy_score(y_test_zcr, y_pred_zcr)
classification_rep_zcr = classification_report(y_test_zcr, y_pred_zcr, zero_division=0)


# Extract RMS Energy features and prepare the dataset
features_rms = []
valid_labels_rms = []

for file, label in zip(audio_files, labels):
    if os.path.isfile(file):
        feature = extract_rms_features(file)
        if feature is not None:
            features_rms.append(feature)
            valid_labels_rms.append(label)
    else:
        print(f"File not found: {file}")

X_rms = np.array(features_rms).reshape(-1, 1)
y_rms = np.array(valid_labels_rms)

# Split the dataset into training and testing sets for RMS Energy
X_train_rms, X_test_rms, y_train_rms, y_test_rms = train_test_split(X_rms, y_rms, test_size=0.3, random_state=42)

# Initialize and train the classifier for RMS Energy
classifier_rms = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_rms.fit(X_train_rms, y_train_rms)

# Make predictions on the test set for RMS Energy
y_pred_rms = classifier_rms.predict(X_test_rms)

# Evaluate the classifier for RMS Energy
accuracy_rms = accuracy_score(y_test_rms, y_pred_rms)
classification_rep_rms = classification_report(y_test_rms, y_pred_rms, zero_division=0)


# Extract Mel Spectrogram features and prepare the dataset
features_mel = []
valid_labels_mel = []

for file, label in zip(audio_files, labels):
    if os.path.isfile(file):
        feature = extract_mel_spectrogram_features(file)
        if feature is not None:
            features_mel.append(feature)
            valid_labels_mel.append(label)
    else:
        print(f"File not found: {file}")

X_mel = np.array(features_mel)
y_mel = np.array(valid_labels_mel)

# Split the dataset into training and testing sets for Mel Spectrogram
X_train_mel, X_test_mel, y_train_mel, y_test_mel = train_test_split(X_mel, y_mel, test_size=0.33, random_state=42)

# Initialize and train the classifier for Mel Spectrogram
classifier_mel = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_mel.fit(X_train_mel, y_train_mel)

# Make predictions on the test set for Mel Spectrogram
y_pred_mel = classifier_mel.predict(X_test_mel)

# Evaluate the classifier for Mel Spectrogram
accuracy_mel = accuracy_score(y_test_mel, y_pred_mel)
classification_rep_mel = classification_report(y_test_mel, y_pred_mel, zero_division=0)


# Function to upload and classify a new audio file based on MFCC features
def upload_and_classify_mfcc():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        feature = extract_mfcc_features(file_path)
        if feature is not None:
            feature = feature.reshape(1, -1)  # Reshape for single sample prediction
            prediction = classifier_mfcc.predict(feature)[0]
            result_text_mfcc.set(f"The uploaded file is classified as: {prediction}")
        else:
            result_text_mfcc.set("Error extracting features from the uploaded file.")


# Function to upload and classify a new audio file based on ZCR features
def upload_and_classify_zcr():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        feature = extract_zcr_features(file_path)
        if feature is not None:
            feature = np.array(feature).reshape(1, -1)  # Reshape for single sample prediction
            prediction = classifier_zcr.predict(feature)[0]
            result_text_zcr.set(f"The uploaded file is classified as: {prediction}")
        else:
            result_text_zcr.set("Error extracting features from the uploaded file.")


# Function to upload and classify a new audio file based on RMS Energy features
def upload_and_classify_rms():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        feature = extract_rms_features(file_path)
        if feature is not None:
            feature = np.array(feature).reshape(1, -1)  # Reshape for single sample prediction
            prediction = classifier_rms.predict(feature)[0]
            result_text_rms.set(f"The uploaded file is classified as: {prediction}")
        else:
            result_text_rms.set("Error extracting features from the uploaded file.")


# Function to upload and classify a new audio file based on Mel Spectrogram features
def upload_and_classify_mel():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        feature = extract_mel_spectrogram_features(file_path)
        if feature is not None:
            feature = feature.reshape(1, -1)  # Reshape for single sample prediction
            prediction = classifier_mel.predict(feature)[0]
            result_text_mel.set(f"The uploaded file is classified as: {prediction}")
        else:
            result_text_mel.set("Error extracting features from the uploaded file.")

# Function to open MFCC results window
def open_mfcc_window():
    mfcc_window = Toplevel(window)
    mfcc_window.title("MFCC Feature Extraction Results")
    mfcc_window.configure(bg='pink')

    title_label_mfcc = Label(mfcc_window, text="MFCC Feature Extraction", font=("Arial", 16, "bold"), bg='pink')
    title_label_mfcc.pack(pady=10)

    text_box_mfcc = scrolledtext.ScrolledText(mfcc_window, wrap=tk.WORD, width=80, height=20, bg='light pink')
    text_box_mfcc.pack(padx=10, pady=10)

    results_mfcc = f"Accuracy: {accuracy_mfcc:.2f}\n\nClassification Report:\n{classification_rep_mfcc}"
    text_box_mfcc.insert(tk.END, results_mfcc)

    upload_button_mfcc = tk.Button(mfcc_window, text="Upload and Classify New File", command=upload_and_classify_mfcc, bg='pink')
    upload_button_mfcc.pack(pady=10)

    result_label_mfcc = Label(mfcc_window, textvariable=result_text_mfcc, bg='pink')
    result_label_mfcc.pack(pady=10)

# Function to open ZCR results window
def open_zcr_window():
    zcr_window = Toplevel(window)
    zcr_window.title("Zero Crossing Rate (ZCR) Feature Extraction Results")
    zcr_window.configure(bg='lightblue')

    title_label_zcr = Label(zcr_window, text="Zero Crossing Rate (ZCR) Feature Extraction", font=("Arial", 16, "bold"), bg='lightblue')
    title_label_zcr.pack(pady=10)

    text_box_zcr = scrolledtext.ScrolledText(zcr_window, wrap=tk.WORD, width=80, height=20, bg='lightblue')
    text_box_zcr.pack(padx=10, pady=10)

    results_zcr = f"Accuracy: {accuracy_zcr:.2f}\n\nClassification Report:\n{classification_rep_zcr}"
    text_box_zcr.insert(tk.END, results_zcr)

    upload_button_zcr = tk.Button(zcr_window, text="Upload and Classify New File", command=upload_and_classify_zcr, bg='lightblue')
    upload_button_zcr.pack(pady=10)

    result_label_zcr = Label(zcr_window, textvariable=result_text_zcr, bg='lightblue')
    result_label_zcr.pack(pady=10)

# Function to open RMS Energy results window
def open_rms_window():
    rms_window = Toplevel(window)
    rms_window.title("RMS Energy Feature Extraction Results")
    rms_window.configure(bg='lightgreen')

    title_label_rms = Label(rms_window, text="RMS Energy Feature Extraction", font=("Arial", 16, "bold"), bg='lightgreen')
    title_label_rms.pack(pady=10)

    text_box_rms = scrolledtext.ScrolledText(rms_window, wrap=tk.WORD, width=80, height=20, bg='lightgreen')
    text_box_rms.pack(padx=10, pady=10)

    results_rms = f"Accuracy: {accuracy_rms:.2f}\n\nClassification Report:\n{classification_rep_rms}"
    text_box_rms.insert(tk.END, results_rms)

    upload_button_rms = tk.Button(rms_window, text="Upload and Classify New File", command=upload_and_classify_rms, bg='lightgreen')
    upload_button_rms.pack(pady=10)

    result_label_rms = Label(rms_window, textvariable=result_text_rms, bg='lightgreen')
    result_label_rms.pack(pady=10)




# Function to open Mel Spectrogram results window

def open_mel_window():
    mel_window = Toplevel(window)
    mel_window.title("Mel Spectrogram Feature Extraction Results")
    mel_window.configure(bg='lightyellow')

    title_label_mel = Label(mel_window, text="Mel Spectrogram Feature Extraction", font=("Arial", 16, "bold"), bg='lightyellow')
    title_label_mel.pack(pady=10)

    text_box_mel = scrolledtext.ScrolledText(mel_window, wrap=tk.WORD, width=80, height=20, bg='lightyellow')
    text_box_mel.pack(padx=10, pady=10)

    results_mel = f"Accuracy: {accuracy_mel:.2f}\n\nClassification Report:\n{classification_rep_mel}"
    text_box_mel.insert(tk.END, results_mel)

    upload_button_mel = tk.Button(mel_window, text="Upload and Classify New File", command=upload_and_classify_mel, bg='lightyellow')
    upload_button_mel.pack(pady=10)

    result_label_mel = Label(mel_window, textvariable=result_text_mel, bg='lightyellow')
    result_label_mel.pack(pady=10)

# Function to compare classification results between feature extraction methods
def compare_classification_results():
    accuracy_scores = [accuracy_mfcc, accuracy_zcr, accuracy_rms, accuracy_mel]
    feature_types = ["MFCC", "ZCR", "RMS Energy", "Mel Spectrogram"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_types, y=accuracy_scores)
    plt.title("Classification Accuracy Comparison")
    plt.xlabel("Feature Extraction Method")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to plot and compare classification reports
def compare_classification_reports():
    reports = [classification_rep_mfcc, classification_rep_zcr, classification_rep_rms, classification_rep_mel]
    feature_types = ["MFCC", "ZCR", "RMS Energy", "Mel Spectrogram"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Classification Report Comparison", fontsize=16)

    for i, (report, feature_type) in enumerate(zip(reports, feature_types)):
        axes[i//2, i%2].text(0.5, 0.5, report, fontsize=10, ha='center', va='center')
        axes[i//2, i%2].set_title(f"Feature Type: {feature_type}")
        axes[i//2, i%2].axis('off')

    plt.tight_layout()
    plt.show()




# Create the main Tkinter window
window = tk.Tk()
window.title("Audio Feature Extraction and Classification")
window.configure(bg='purple')
window.geometry("600x500")


# Create buttons to open MFCC, ZCR, RMS Energy, and Mel Spectrogram windows
mfcc_button = tk.Button(window, text="MFCC Feature Extraction Results", command=open_mfcc_window, bg='pink')
mfcc_button.pack(pady=20)

zcr_button = tk.Button(window, text="ZCR Feature Extraction Results", command=open_zcr_window, bg='lightblue')
zcr_button.pack(pady=20)

rms_button = tk.Button(window, text="RMS Energy Feature Extraction Results", command=open_rms_window, bg='lightgreen')
rms_button.pack(pady=20)

mel_button = tk.Button(window, text="Mel Spectrogram Feature Extraction Results", command=open_mel_window, bg='lightyellow')
mel_button.pack(pady=20)

# Text variables to display the results of classification
result_text_mfcc = tk.StringVar()
result_text_zcr = tk.StringVar()
result_text_rms = tk.StringVar()
result_text_mel = tk.StringVar()

# Create button to plot and compare classification results
compare_button = ttk.Button(window, text="Compare Classification Results", command=compare_classification_results)
compare_button.pack(pady=20)

# Create button to plot and compare classification reports
compare_reports_button = ttk.Button(window, text="Compare Classification Reports", command=compare_classification_reports)
compare_reports_button.pack(pady=20)


# Start the Tkinter event loop
window.mainloop()
