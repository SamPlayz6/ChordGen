import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
import numpy as np
import random

# Helper function to read data from CSV files
def read_from_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(' '.join(row))  # Merge all columns into a single string per row
    return data

# Tokenization and mapping functions
def create_tokenizer(data):
    tokens = set()
    for seq in data:
        tokens.update(seq.split())
    token_to_id = {token: idx + 1 for idx, token in enumerate(tokens)}
    token_to_id['PAD'] = 0
    token_to_id['UNK'] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return lambda seq: [token_to_id.get(token, token_to_id['UNK']) for token in seq.split()], token_to_id, id_to_token

# Load data function to handle melodies and chords
def load_data(melody_file, chord_file):
    melody_data = read_from_csv(melody_file)
    chord_data = read_from_csv(chord_file)
    tokenizer_melody, token_to_id_melody, id_to_token_melody = create_tokenizer(melody_data)
    tokenizer_chord, token_to_id_chord, id_to_token_chord = create_tokenizer(chord_data)
    tokenized_melodies = [tokenizer_melody(sequence) for sequence in melody_data]
    tokenized_chords = [tokenizer_chord(sequence) for sequence in chord_data]
    return tokenized_melodies, tokenized_chords, token_to_id_melody, id_to_token_melody, token_to_id_chord, id_to_token_chord

class MelodyChordDataset(Dataset):
    def __init__(self, melodies, chords, max_length=None):
        self.melodies = melodies
        self.chords = chords
        self.max_length = max_length if max_length is not None else max(len(m) for m in melodies + chords)

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        melody = torch.tensor(self.melodies[idx], dtype=torch.long)
        chord = torch.tensor(self.chords[idx], dtype=torch.long)
        if self.max_length:
            # Pad sequences to the maximum length
            melody = pad(melody, (0, self.max_length - len(melody)), value=0)
            chord = pad(chord, (0, self.max_length - len(chord)), value=0)
        return melody, chord
    

# Define LSTM model classes
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim  # Define the output dimension attribute
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for i, (melodies, chords) in enumerate(train_loader):
            melodies, chords = melodies.to(device), chords.to(device)
            optimizer.zero_grad()
            outputs = model(melodies, chords)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), chords.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            total += chords.nelement()
            correct += (predicted == chords).sum().item()

            if i % 4 == 0:  # Adjust this depending on the size of your dataset
                print(f'Batch {i}, Loss: {loss.item()}, Acc: {100 * correct / total:.2f}%')

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for melodies, chords in val_loader:
            melodies, chords = melodies.to(device), chords.to(device)
            outputs = model(melodies, chords)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), chords.view(-1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            total += chords.nelement()
            correct += (predicted == chords).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc




def predict(model, input_sequence, token_to_id, id_to_token, device):
    model.eval()
    input_tensor = torch.tensor([[token_to_id.get(note, token_to_id['UNK']) for note in input_sequence.split(',')]], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor, torch.zeros((1, input_tensor.size(1)), dtype=torch.long).to(device))
        predicted_indices = output.argmax(2).squeeze().tolist()
        predicted_chords = [id_to_token[idx] for idx in predicted_indices]
    return predicted_chords

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_melodies, tokenized_chords, token_to_id_melody, id_to_token_melody, token_to_id_chord, id_to_token_chord = load_data('data/melodies.csv', 'data/chords.csv')

    dataset = MelodyChordDataset(tokenized_melodies, tokenized_chords)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32)  # Placeholder for actual validation data

    model = Seq2Seq(
        Encoder(input_dim=len(token_to_id_melody), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.5),
        Decoder(output_dim=len(token_to_id_chord), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.5),
        device
    ).to(device)

    if args.mode == 'train':
        train_model(model, train_loader, val_loader, 10, 0.001, device)  # 10 epochs and learning rate of 0.001
        torch.save(model.state_dict(), 'model.pth')
    elif args.mode == 'inference':
        model.load_state_dict(torch.load('model.pth'))
        input_sequence = "C0/6,C1/6,C0/6,B0/5,A0/5,G0/5,F1/5,D0/5,F1/5,E0/5,B0/5,F0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,D0/5,D0/5A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,A0/5,A0/5,F1/5,G0/5,D0/5,D0/5,F1/5,A0/5,C1/6,E0/6,D0/6,D0/5,C1/5,D0/5,E0/6,E0/6,D0/6,B0/5,A0/5,F1/5,"
        predicted_chords = predict(model, input_sequence, token_to_id_melody, id_to_token_melody, device)
        print("Predicted Chords:", predicted_chords)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or run inference on an LSTM model for music generation.')
    parser.add_argument('mode', choices=['train', 'inference'], help='Choose "train" or "inference" mode')
    # parser.add_argument('--input_melody', type=str, help='Input melody for inference, formatted as comma-separated notes') # Here later I can get it to maybe look at where the sung audio file is saved
    args = parser.parse_args()
    main(args)
