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

