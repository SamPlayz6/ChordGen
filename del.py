import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('singing_voice.wav')

# Extract the melody
pitch, mag = librosa.piptrack(y=y, sr=sr)

# Identify the most dominant pitch at each time step
melody = pitch[np.argmax(mag, axis=0), np.arange(pitch.shape[1])]

# Plot the melody
plt.figure(figsize=(12, 6))
plt.plot(melody)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Extracted Melody')
plt.show()


#Example Above of data processing + Extraction - Does not include pitch correction or removal of noise


