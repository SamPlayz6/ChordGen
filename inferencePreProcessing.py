import crepe
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def analyze_pitch(filename):
    # Load the audio file
    sr, audio = scipy.io.wavfile.read(filename)
    if audio.ndim > 1:  # Convert stereo to mono if necessary
        audio = audio.mean(axis=1)

    # Run CREPE on the audio file
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    # Filter out low-confidence detections
    mask = confidence > 0.7
    time, frequency = time[mask], frequency[mask]

    return time, frequency, confidence

def plot_pitch(time, frequency):
    plt.figure(figsize=(10, 4))
    plt.plot(time, frequency, label="Pitch")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Detection using CREPE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filename = "SungMelodies/M1.wav"
    time, frequency, confidence = analyze_pitch(filename)
    plot_pitch(time, frequency)