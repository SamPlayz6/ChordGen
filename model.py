import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Function to read data from a CSV file
def read_from_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Tokenize and pad sequences
def tokenize_and_pad(data, tokenizer, max_len):
    sequences = [tokenizer(seq) for seq in data]
    padded_sequences = [seq[:max_len] + [0] * (max_len - len(seq)) for seq in sequences]  # Ensure padding
    return np.array(padded_sequences, dtype=np.int32)  # Specify dtype to avoid VisibleDeprecationWarning

# Create a simple tokenizer with UNK handling
def create_tokenizer(data):
    tokens = set()
    for seq in data:
        for item in seq:
            tokens.add(item)
    token_to_id = {token: idx + 1 for idx, token in enumerate(tokens)}
    token_to_id['PAD'] = 0
    token_to_id['UNK'] = len(token_to_id) + 1  # Adding UNK token
    return lambda seq: [token_to_id.get(token, token_to_id['UNK']) for token in seq], token_to_id

# Custom PyTorch Dataset
class MelodyChordDataset(Dataset):
    def __init__(self, melodies, chords):
        self.melodies = melodies
        self.chords = chords

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        melody = torch.tensor(self.melodies[idx], dtype=torch.long)
        chord = torch.tensor(self.chords[idx], dtype=torch.long)
        return melody, chord[-1]  # Return only the last chord for sequence-to-one prediction

# Read the saved CSV files
melodies = read_from_csv('data/melodies.csv')
chords = read_from_csv('data/chords.csv')

# Flatten the nested lists
flat_melodies = [item for sublist in melodies for item in sublist]
flat_chords = [item for sublist in chords for item in sublist]

# Create tokenizers for melodies and chords
melody_tokenizer, melody_token_to_id = create_tokenizer(flat_melodies)
chord_tokenizer, chord_token_to_id = create_tokenizer(flat_chords)

# Ensure input_size includes all token indices
input_size = max(len(melody_token_to_id), len(chord_token_to_id)) + 1  # Add 1 to cover all indices

# Define maximum sequence length (adjust as necessary)
max_len = 100

# Tokenize and pad melodies and chords
tokenized_melodies = tokenize_and_pad(melodies, melody_tokenizer, max_len)
tokenized_chords = tokenize_and_pad(chords, chord_tokenizer, max_len)

# Create the dataset
dataset = MelodyChordDataset(tokenized_melodies, tokenized_chords)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# LSTM Model definition with Embedding Layer
class LSTMModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for melodies, chords in train_loader:
            melodies = melodies.to(device)
            chords = chords.to(device)

            outputs = model(melodies)
            loss = criterion(outputs, chords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for melodies, chords in val_loader:
                melodies = melodies.to(device)
                chords = chords.to(device)

                outputs = model(melodies)
                loss = criterion(outputs, chords)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
embedding_dim = 128  # Dimension of the embedding layer
hidden_size = 256  # Number of LSTM units
output_size = len(chord_token_to_id) + 1  # Size of chord vocabulary
num_layers = 4  # Number of LSTM layers
dropout = 0.5  # Dropout rate
num_epochs = 15  # Number of epochs
learning_rate = 0.0005  # Learning rate

# Initialize the model, move to device and start training
model = LSTMModel(input_size, embedding_dim, hidden_size, output_size, num_layers, dropout).to(device)
train_model(model, train_loader, val_loader, num_epochs, learning_rate)