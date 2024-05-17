import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Function to parse an XML file and extract melody and chords
def parse_xml_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    namespace = {'score': 'http://www.musicxml.org/dtds/partwise.dtd'}

    melodies = []
    chords = []

    for part in root.findall('score:part', namespace):
        for measure in part.findall('score:measure', namespace):
            for note in measure.findall('score:note', namespace):
                pitch = note.find('score:pitch', namespace)
                if pitch is not None:
                    step = pitch.find('score:step', namespace).text
                    octave = pitch.find('score:octave', namespace).text
                    alter = pitch.find('score:alter', namespace)
                    alter = alter.text if alter is not None else '0'
                    melodies.append(f"{step}{alter}/{octave}")
            
            for harmony in measure.findall('score:harmony', namespace):
                root = harmony.find('score:root', namespace)
                kind = harmony.find('score:kind', namespace)
                if root is not None and kind is not None:
                    root_step = root.find('score:root-step', namespace).text
                    root_alter = root.find('score:root-alter', namespace)
                    root_alter = root_alter.text if root_alter is not None else '0'
                    chords.append(f"{root_step}{root_alter}/{kind.text}")

    return melodies, chords

# Function to load all data from the dataset directory
def load_and_preprocess_data(dataset_directory):
    all_melodies = []
    all_chords = []

    for subdir, _, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith('.xml'):
                filepath = os.path.join(subdir, file)
                melodies, chords = parse_xml_file(filepath)
                all_melodies.append(melodies)
                all_chords.append(chords)

    return all_melodies, all_chords

# Example dataset directory
dataset_directory = 'path_to_your_chord_melody_dataset'
melodies, chords = load_and_preprocess_data(dataset_directory)

# Function to prepare the dataset for PyTorch
def prepare_dataset(melodies, chords):
    # Assuming a dummy function to convert lists to tensors - replace with actual preprocessing
    melody_tensor = torch.tensor([list(map(ord, melody)) for melody in melodies], dtype=torch.long)
    chord_tensor = torch.tensor([list(map(ord, chord)) for chord in chords], dtype=torch.long)
    dataset = TensorDataset(melody_tensor, chord_tensor)
    return dataset

dataset = prepare_dataset(melodies, chords)




#==================================


# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# LSTM Model definition
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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
input_size = 128  # Example, need to adjust according to actual data
hidden_size = 128
output_size = 128  # Example, need to adjust according to actual data
num_layers = 2
num_epochs = 20
learning_rate = 0.001

# Initialize the model, move to device and start training
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
train_model(model, train_loader, val_loader, num_epochs, learning_rate)