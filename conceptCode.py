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
import torch


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
    # Flatten the chords list and get unique notes
    unique_notes = list(set(melody + [note for chord in chords for note in chord]))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    
    # Encode the melody
    melody_encoded = [note_to_int[note] for note in melody]
    
    # Encode the chords
    chords_encoded = [[note_to_int[note] for note in chord] for chord in chords]
    
    return melody_encoded, chords_encoded

def one_hot_encode(melody, chords):
    # Flatten the chords list and get unique notes
    unique_notes = list(set(melody + [note for chord in chords for note in chord]))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    
    # Encode the melody
    melody_encoded = [note_to_int[note] for note in melody]
    
    # One-hot encode the melody
    melody_one_hot = torch.nn.functional.one_hot(torch.tensor(melody_encoded), num_classes=len(unique_notes))
    
    # Encode and one-hot encode the chords
    chords_encoded = [[note_to_int[note] for note in chord] for chord in chords]
    chords_one_hot = [torch.nn.functional.one_hot(torch.tensor(chord), num_classes=len(unique_notes)) for chord in chords_encoded]
    
    return melody_one_hot, chords_one_hot



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
                    chords.append([note.pitch for note in instrument.notes[i - 1::-1] if note.end > instrument.notes[i - 1].start])

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



-----------------
# Model structure

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])

        return out





# Training and splitting

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have input data `inputs` and target data `targets`
inputs = torch.randn(total_samples, sequence_length, input_size).to(device)
targets = torch.randint(0, 2, (total_samples,)).to(device)  # Random binary targets

# Split the data into training and validation sets
inputs_np = inputs.cpu().numpy()
targets_np = targets.cpu().numpy()
inputs_train_np, inputs_val_np, targets_train_np, targets_val_np = train_test_split(inputs_np, targets_np, test_size=0.2)
inputs_train = torch.tensor(inputs_train_np).to(device)
inputs_val = torch.tensor(inputs_val_np).to(device)
targets_train = torch.tensor(targets_train_np).to(device)
targets_val = torch.tensor(targets_val_np).to(device)

# Initialize network
model = LSTMNet(input_size, hidden_size, output_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(inputs_train)
    loss = criterion(outputs, targets_train)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        outputs_val = model(inputs_val)
        predictions = torch.round(torch.sigmoid(outputs_val))  # Sigmoid to get the probabilities and rounding to get the class
        accuracy = accuracy_score(targets_val.cpu().numpy(), predictions.cpu().numpy())

    if (epoch+1) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, loss.item(), accuracy))









