import os
import csv
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Function to parse an XML file and extract melody and chords
def parse_xml_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    melodies = []
    chords = []

    for part in root.findall('part'):
        for measure in part.findall('measure'):
            for note in measure.findall('note'):
                pitch = note.find('pitch')
                if pitch is not None:
                    step = pitch.find('step').text
                    octave = pitch.find('octave').text
                    alter = pitch.find('alter')
                    alter = alter.text if alter is not None else '0'
                    melodies.append(f"{step}{alter}/{octave}")
            
            for harmony in measure.findall('harmony'):
                root = harmony.find('root')
                kind = harmony.find('kind')
                if root is not None and kind is not None:
                    root_step = root.find('root-step').text
                    root_alter = root.find('root-alter')
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
                
                # Debugging file processing
                print(f"Processing file: {filepath}")  # Debug statement
                
                melodies, chords = parse_xml_file(filepath)
                
                # Debugging data extraction
                print(f"Extracted {len(melodies)} melodies and {len(chords)} chords")  # Debug statement
                
                if melodies:  # Only add non-empty lists
                    all_melodies.append(melodies)
                if chords:  # Only add non-empty lists
                    all_chords.append(chords)

    return all_melodies, all_chords


# Save data to CSV
def save_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    print(f"Saved {len(data)} records to {filename}")  # Debug statement

# Example dataset directory
dataset_directory = 'c:\\Users\\user\\Documents\\GitHub\\ChordGen\\chord-melody-dataset-master'
melodies, chords = load_and_preprocess_data(dataset_directory)

# Save melodies and chords to CSV files
save_to_csv(melodies, 'data/melodies.csv')
save_to_csv(chords, 'data/chords.csv')
