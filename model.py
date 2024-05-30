import argparse
import json
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Helper functions
def read_from_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def tokenize_and_pad(data, tokenizer, max_len):
    sequences = [tokenizer(seq) for seq in data]
    padded_sequences = [seq[:max_len] + [0] * (max_len - len(seq)) for seq in sequences]
    return np.array(padded_sequences, dtype=np.int32)

def create_tokenizer(data):
    tokens = set()
    for seq in data:
        tokens.update(seq)
    token_to_id = {token: idx + 1 for idx, token in enumerate(tokens)}
    token_to_id['PAD'] = 0
    token_to_id['UNK'] = len(token_to_id) + 1
    return lambda seq: [token_to_id.get(token, token_to_id['UNK']) for token in seq], token_to_id

class MelodyChordDataset(Dataset):
    def __init__(self, melodies, chords):
        self.melodies = melodies
        self.chords = chords

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        melody = torch.tensor(self.melodies[idx], dtype=torch.long)
        chord = torch.tensor(self.chords[idx], dtype=torch.long)
        return melody, chord[-1]

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

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout):
    model = LSTMModel(input_size, embedding_dim, hidden_size, output_size, num_layers, dropout)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, tokenizer, input_sequence, max_len, chord_token_to_id):
    tokenized_input = tokenize_and_pad([input_sequence], tokenizer, max_len)
    input_tensor = torch.tensor(tokenized_input, dtype=torch.long).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = output.argmax(1).item()
    id_to_token = {v: k for k, v in chord_token_to_id.items()}
    return id_to_token[predicted_index]

def save_configurations(path, input_size, chord_token_to_id):
    with open(path, 'w') as f:
        json.dump({'input_size': input_size, 'chord_token_to_id': chord_token_to_id}, f)

def load_configurations(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config['input_size'], {int(k): v for k, v in config['chord_token_to_id'].items()}

def main(args):
    config_path = 'Example/model_config.json'  # Path to save configuration
    save_model_path = 'Example/lstm_model.pth'  # Path to save the model
    
    if args.mode == 'train':
        print("Starting training...")
        melodies = read_from_csv('data/melodies.csv')
        chords = read_from_csv('data/chords.csv')
        flat_melodies = [item for sublist in melodies for item in sublist]
        flat_chords = [item for sublist in chords for item in sublist]
        melody_tokenizer, melody_token_to_id = create_tokenizer(flat_melodies)
        chord_tokenizer, chord_token_to_id = create_tokenizer(flat_chords)
        input_size = max(len(melody_token_to_id), len(chord_token_to_id)) + 1
        max_len = 100
        tokenized_melodies = tokenize_and_pad(melodies, melody_tokenizer, max_len)
        tokenized_chords = tokenize_and_pad(chords, chord_tokenizer, max_len)
        dataset = MelodyChordDataset(tokenized_melodies, tokenized_chords)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size, 128, 256, len(chord_token_to_id) + 1, 3, 0.5).to(device)
        train_model(model, train_loader, val_loader, 15, 0.0005, device)
        save_model(model, save_model_path)
        
        # Save the configuration
        save_configurations(config_path, input_size, chord_token_to_id)

    elif args.mode == 'inference':
        print("Starting inference...")
        
        # Load the configuration
        input_size, chord_token_to_id = load_configurations(config_path)
        
        loaded_model = load_model(save_model_path, input_size, 128, 256, len(chord_token_to_id) + 1, 3, 0.5)
        example_input_sequence = "A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,A0/5,A0/5,F1/5,G0/5,D0/5,D0/5,F1/5,A0/5,C1/6,E0/6,D0/6,D0/5,C1/5,D0/5,E0/6,E0/6,D0/6,B0/5,A0/5,F1/5,C0/6,C1/6,C0/6,B0/5,A0/5,G0/5,F1/5,D0/5,F1/5,E0/5,B0/5,F0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,D0/5,D0/5"
        predicted_chord = predict(loaded_model, melody_tokenizer, example_input_sequence, max_len, chord_token_to_id)
        print("Predicted Chord:", predicted_chord)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or run inference on an LSTM model.')
    parser.add_argument('mode', choices=['train', 'inference'], help='Select mode to either "train" or run "inference"')
    args = parser.parse_args()
    main(args)