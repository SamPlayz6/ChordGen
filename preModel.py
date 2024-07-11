import os
import argparse
from music21 import converter
from mido import MidiFile

#--------------------XML to MIDI(Not Part Of Flow)-------------------

def convert_xml_to_midi(xml_file_path, midi_file_path):
    # Load the MusicXML file
    score = converter.parse(xml_file_path)
    
    # Save as MIDI
    midi = score.write('midi', fp=midi_file_path)
    print(f"MIDI file saved as {midi_file_path}")



#--------------MIDI to String(Of notes)-------------

def midi_to_note_string(midi_file, ticks_per_beat=480):
    midi = MidiFile(midi_file)
    note_events = []
    current_time = 0
    note_string = []

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append((msg.note, current_time))
            if not msg.is_meta:
                current_time += msg.time

    # Convert note numbers to note names
    note_names = ['C0', 'C1', 'D0', 'D1', 'E0', 'F0', 'F1', 'G0', 'G1', 'A0', 'A1', 'B0']
    for note, time in note_events:
        note_name = note_names[note % 12]
        octave = note // 12 - 1  # MIDI note 60 is middle C (C4), which corresponds to note_name[0] in octave 4
        note_string.append(f"{note_name}/{octave}")

    return ",".join(note_string)


#-----------------AudioToMIDI(Or StringOfNotes)------------
import librosa
import numpy as np
from midiutil import MIDIFile

def wav_to_midi(input_wav, output_midi, min_duration=0.1):
    # Load the audio file
    y, sr = librosa.load(input_wav)
    
    # Get total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Use pYIN for improved pitch detection
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'))

    # Create a MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, 120)

    # Set the instrument to Acoustic Grand Piano
    program = 0
    midi.addProgramChange(track, 0, time, program)

    # Process the pitch data
    last_pitch = None
    start_time = 0
    hop_time = total_duration / len(f0)

    for i, pitch in enumerate(f0):
        current_time = i * hop_time
        if voiced_flag[i] and pitch is not None:
            midi_pitch = round(librosa.hz_to_midi(pitch))
            if midi_pitch != last_pitch:
                if last_pitch is not None:
                    duration = current_time - start_time
                    if duration >= min_duration:
                        velocity = min(127, max(1, int(voiced_probs[i] * 127)))
                        midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                start_time = current_time
                last_pitch = midi_pitch
        else:
            if last_pitch is not None:
                duration = current_time - start_time
                if duration >= min_duration:
                    velocity = min(127, max(1, int(voiced_probs[i-1] * 127)))
                    midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                last_pitch = None

    # Add the last note if there is one
    if last_pitch is not None:
        duration = total_duration - start_time
        if duration >= min_duration:
            velocity = min(127, max(1, int(voiced_probs[-1] * 127)))
            midi.addNote(track, 0, last_pitch, start_time, duration, velocity)

    # Save the MIDI file
    with open(output_midi, "wb") as output_file:
        midi.writeFile(output_file)


if __name__ == "__main__":
    # try:
    #     file_path, file_type = get_single_file_from_directory('Input/')
    #     print("test1")
    # except Exception as e:
    #     print(e)
    #     print("test2")

    parser = argparse.ArgumentParser(description='Input File Path + Name')
    parser.add_argument('file_path', nargs='?', help='State file path for file to be processed')

    args = parser.parse_args()


    if args.file_path[-4:] == ".wav":
        print("Processing .wav file...")
        # Input - Output paths
        wav_to_midi(args.file_path, 'Input/tempStorage/ModelInputMIDI.mid')
        # Converting Melody MIDI to Melody String Sequence
        note_string = midi_to_note_string('Input/tempStorage/ModelInputMIDI.mid')
        print(note_string)


    elif args.file_path[-4:] == ".mid":
        print("Processing .mid file..")
        # Converting Melody MIDI to Melody String Sequence
        note_string = midi_to_note_string(args.file_path)
        print(note_string)

    else:
        print("Input file is neither .wav or .mid",args.file_path[-4:])

