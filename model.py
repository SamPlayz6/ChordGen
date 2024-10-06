import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

# Helper function to read data from CSV files
def read_from_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.extend(row)  # Append each cell to a single list
    return ' '.join(data)  # Return a single string containing all data

def create_tokenizer(data):
    # Assuming data is a single string with space-separated tokens
    tokens = set(data.split())
    # Additional handling for complex chord structures
    token_to_id = {token: idx + 1 for idx, token in enumerate(sorted(tokens))}
    token_to_id['PAD'] = 0
    token_to_id['UNK'] = max(token_to_id.values()) + 1  # Ensuring unique index for 'UNK'
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return lambda seq: [token_to_id.get(token, token_to_id['UNK']) for token in seq.split()], token_to_id, id_to_token

def create_sequences(tokenized_data, sequence_length, step):
    sequences = []
    # tokenized_data now should be a single long list of tokens
    for i in range(0, len(tokenized_data) - sequence_length + 1, step):
        sequences.append(tokenized_data[i:i+sequence_length])
    return sequences

def load_data(melody_file, chord_file, sequence_length, step):
    melody_data = read_from_csv(melody_file)
    chord_data = read_from_csv(chord_file)
    tokenizer_melody, token_to_id_melody, id_to_token_melody = create_tokenizer(melody_data)
    tokenizer_chord, token_to_id_chord, id_to_token_chord = create_tokenizer(chord_data)

    tokenized_melodies = tokenizer_melody(melody_data)
    tokenized_chords = tokenizer_chord(chord_data)
    tokenized_melodies = create_sequences(tokenized_melodies, sequence_length, step)
    tokenized_chords = create_sequences(tokenized_chords, sequence_length, step)

    if not tokenized_melodies or not tokenized_chords:
        raise ValueError("No data sequences were generated. Check the input data and parameters.")

    return tokenized_melodies, tokenized_chords, token_to_id_melody, id_to_token_melody, token_to_id_chord, id_to_token_chord

class MelodyChordDataset(Dataset):
    def __init__(self, melodies, chords):
        assert len(melodies) == len(chords), "Melodies and chords must have the same number of entries."
        self.melodies = melodies
        self.chords = chords

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        melody = torch.tensor(self.melodies[idx], dtype=torch.long)
        chord = torch.tensor(self.chords[idx], dtype=torch.long)
        return melody, chord

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

            if i % 4 == 0:
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

def predict(model, input_sequence, token_to_id_melody, id_to_token_chord, device, temperature=1.0):
    model.eval()
    input_tensor = torch.tensor([[token_to_id_melody.get(note, token_to_id_melody['UNK']) for note in input_sequence.split(',')]], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor, torch.zeros((1, input_tensor.size(1)), dtype=torch.long).to(device))
        output = output / temperature
        probs = torch.nn.functional.softmax(output, dim=-1)
        predicted_indices = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1]).tolist()
        predicted_chords = [id_to_token_chord.get(idx, 'UNK') for idx in predicted_indices[0]]
    return predicted_chords

def main(mode, input_sequence=None, sequence_length=30, step=5, batch_size=32, num_epochs=10, learning_rate=0.001, emb_dim=256, hid_dim=256, n_layers=2, dropout=0.5, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_melodies, tokenized_chords, token_to_id_melody, id_to_token_melody, token_to_id_chord, id_to_token_chord = load_data(
        'data/TrainingData/melodies.csv',
        'data/TrainingData/expandedChords.csv',
        sequence_length,
        step
    )

    dataset = MelodyChordDataset(tokenized_melodies, tokenized_chords)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)  # Placeholder for actual validation data

    model = Seq2Seq(
        Encoder(input_dim=len(token_to_id_melody), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers, dropout=dropout),
        Decoder(output_dim=len(token_to_id_chord), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers, dropout=dropout),
        device
    ).to(device)

    if mode == 'train':
        train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
        torch.save(model.state_dict(), 'Input/Misc/model.pth')
    elif mode == 'inference':
        model.load_state_dict(torch.load('Input/Misc/model.pth'))
        predicted_chords = predict(model, input_sequence, token_to_id_melody, id_to_token_chord, device, temperature)
        print(','.join(predicted_chords))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or run inference on an LSTM model for music generation.')
    parser.add_argument('mode', choices=['train', 'inference'], help='Choose "train" or "inference" mode')
    parser.add_argument('input_sequence', nargs='?', help='Additional input required for inference mode')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling during inference')

    args = parser.parse_args()

    # Changable Parameters
    sequence_length = 30  # Length of each input sequence
    step = 5  # Step size for sliding window
    batch_size = 32  # Batch size for DataLoader
    num_epochs = 10  
    learning_rate = 0.001  
    emb_dim = 256  
    hid_dim = 256  
    n_layers = 2  
    dropout = 0.5  

    # Depending on the mode, call main with different parameters
    if args.mode == 'train':
        main(args.mode, sequence_length=sequence_length, step=step, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers, dropout=dropout)
    elif args.mode == 'inference':
        if not args.input_sequence:
            parser.error("Inference mode requires an additional input.")
        main(args.mode, args.input_sequence, sequence_length=sequence_length, step=step, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers, dropout=dropout, temperature=args.temperature)
