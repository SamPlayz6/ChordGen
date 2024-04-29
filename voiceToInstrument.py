import librosa
import matplotlib.pyplot as plt
import numpy as np


y, sr = librosa.load('C:\\Users\\user\\Documents\\GitHub\\ChordGen\\SungMelodies\\M1.wav',   sr = 22050, mono = True )
# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

pitches, magnitudes = librosa.piptrack(y=y, sr=sr)




print(type(pitches[0][0]),pitches[100])
print(magnitudes[100])



#melspectrogram 
# D = np.abs(librosa.stft(y))**2
# S = librosa.feature.melspectrogram(S=D, sr=sr)

# librosa.feature.melspectrogram(y=y, sr=sr)
# fig, ax = plt.subplots()
# s_db = librosa.power_to_db(S, ref=np.max)
# img = librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sr,fmax=8000, ax=ax)

# fig.colorbar(img, ax=ax, format= '%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# plt.show()