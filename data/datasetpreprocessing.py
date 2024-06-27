import os
import csv
import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy

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
                    alter = int(alter.text) if alter is not None else 0
                    note_name = f"{step}{alter}/{octave}"
                    melodies.append(note_name)
            
            for harmony in measure.findall('harmony'):
                root = harmony.find('root')
                kind = harmony.find('kind')
                if root is not None and kind is not None:
                    root_step = root.find('root-step').text
                    root_alter = root.find('root-alter')
                    root_alter = int(root_alter.text) if root_alter is not None else 0
                    chord_name = f"{root_step}{root_alter}/{kind.text}"
                    chords.append(chord_name)

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
                # print(f"Processing file: {filepath}")  # Debug statement
                
                melodies, chords = parse_xml_file(filepath)
                
                # Debugging data extraction
                # print(f"Extracted {len(melodies)} melodies and {len(chords)} chords")  # Debug statement
                
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

# Function to transpose melodies by a given number of semitones
def transpose_melody(melody, semitones):
    transposed_melody = []
    step_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 
        'B': 11, 'Cb': 11, 'B#': 0,
        'B-': 10, 'A-': 8, 'E-': 3, 'F-': 4, 'C-': 11, 'D-': 1, 'G-': 6
    }
    reverse_step_map = {v: k for k, v in step_map.items()}
    
    for note in melody:
        if '/' in note:
            step_alter, octave = note.split('/')
            step = step_alter[:-1] if len(step_alter) > 1 else step_alter[0]
            alter = int(step_alter[-1]) if len(step_alter) > 1 else 0
            
            if step not in step_map:
                print(f"Warning: Unrecognized note step '{step}'. Skipping note.")
                transposed_melody.append(note)
                continue

            original_pitch = (step_map[step] + alter) % 12
            transposed_pitch = (original_pitch + semitones) % 12
            transposed_step = reverse_step_map[transposed_pitch]
            transposed_note = f"{transposed_step}/{octave}"
            transposed_melody.append(transposed_note)
        else:
            transposed_melody.append(note)
    
    return transposed_melody

# Function to augment symbolic data by transposing melodies
def augment_symbolic_data(melodies, chords):
    augmented_melodies = []
    augmented_chords = []

    for melody, chord in zip(melodies, chords):
        for semitones in range(-4, 5):  # Transpose by -4 to +4 semitones
            if semitones == 0:
                continue  # Skip original melody
            transposed_melody = transpose_melody(melody, semitones)
            augmented_melodies.append(transposed_melody)
            augmented_chords.append(deepcopy(chord))

    return augmented_melodies, augmented_chords



# Filling in the chords to the chords dataset to increase to the same length as the notes dataset
import csv

def read_csv_to_string(filename):
    """ Read a CSV file and convert contents to a single string. """
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data

def expand_chords(chords_data, notes_count):
    """ Repeat each chord to match the length of the notes data. """
    total_chords = sum(len(line) for line in chords_data)
    repeats_per_chord = notes_count // total_chords
    extra = notes_count % total_chords

    expanded_chords = []
    for line in chords_data:
        for chord in line:
            expanded_chords.extend([chord] * repeats_per_chord)
            if extra > 0:
                expanded_chords.append(chord)
                extra -= 1

    return ','.join(expanded_chords)  # Joining with commas for CSV format

def process_chords(melodies, chords, expandedChords):
    melody_data = read_csv_to_string(melodies)
    chords_data = read_csv_to_string(chords)

    # Flatten the melody_data to calculate the total number of notes
    flat_melody_data = [item for sublist in melody_data for item in sublist]
    notes_count = len(flat_melody_data)

    expanded_chords = expand_chords(chords_data, notes_count).split(',')  # Splitting into a list

    with open(expandedChords, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(expanded_chords)  # Writing as separate fields

# Example usage - 'path_to_notes.csv', 'path_to_chords.csv', 'output_chords.csv'
process_chords('data/melodies.csv', 'data/chords.csv', 'data/expandedChords.csv')





# # Example dataset directory
# dataset_directory = 'c:\\Users\\user\\Documents\\GitHub\\ChordGen\\chord-melody-dataset-master'
# melodies, chords = load_and_preprocess_data(dataset_directory)

# # Augment the data
# augmented_melodies, augmented_chords = augment_symbolic_data(melodies, chords)

# # Combine original and augmented data
# combined_melodies = melodies + augmented_melodies
# combined_chords = chords + augmented_chords

# # Save combined data to new CSV files
# save_to_csv(melodies, 'data/melodies.csv')
# save_to_csv(chords, 'data/chords.csv')