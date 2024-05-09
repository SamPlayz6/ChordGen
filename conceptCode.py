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


#------------------------

import os
import pretty_midi
import numpy as np


def transpose_to_c_major(pm):
    key = pm.key_signature_changes
    if len(key) > 0:
        key = key[0].key_number
    else:
        key = 0
    semitones = -key
    for instrument in pm.instruments:
        for note in instrument.notes:
            note.pitch += semitones
    return pm

def encode_melody_and_chords(melody, chords):
    unique_notes = list(set(melody))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    melody_encoded = [note_to_int[note] for note in melody]

    unique_chords = list(set(chords))
    chord_to_int = {chord: i for i, chord in enumerate(unique_chords)}
    chords_encoded = [chord_to_int[chord] for chord in chords]

    return melody_encoded, chords_encoded



def extract_melody_and_chords(pm):
    melody = []
    chords = []
    for instrument in pm.instruments:
        # Sort the notes by their start time
        instrument.notes.sort(key=lambda note: note.start)
        for i in range(len(instrument.notes)):
            if i == 0 or instrument.notes[i].start > instrument.notes[i - 1].end:
                # This note starts a new chord
                melody.append(instrument.notes[i].pitch)
                if i > 0:
                    # Append the previous chord to the chords list
                    chords.append([note.pitch for note in instrument.notes[i - 1::-1] if note.end > instrument.notes[i - 1].start]

                else:
                # This note is part of the current chord
                instrument.notes[i - 1].end = max(instrument.notes[i - 1].end, instrument.notes[i].end)
        # Append the last chord to the chords list
        chords.append([note.pitch for note in instrument.notes[::-1] if note.end > instrument.notes[-1].start])
    return melody, chords




# Directory where the Lakh MIDI dataset is stored
midi_dir = 'path_to_your_midi_files'

# Iterate over all MIDI files in the directory
for filename in os.listdir(midi_dir):
    if filename.endswith('.mid'):
        # Load the MIDI file
        pm = pretty_midi.PrettyMIDI(os.path.join(midi_dir, filename))

        # Extract melody and chords
        melody, chords = extract_melody_and_chords(pm)

        # Transpose to C major
        pm = transpose_to_c_major(pm)

    # Encode melody and chords
        melody_encoded, chords_encoded = encode_melody_and_chords(melody, chords)

        # Now melody_encoded and chords_encoded can be used as input and output for your LSTM model






