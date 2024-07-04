from music21 import converter
from mido import MidiFile

#--------------------XML to MIDI(Not Part Of Flow)-------------------

def convert_xml_to_midi(xml_file_path, midi_file_path):
    # Load the MusicXML file
    score = converter.parse(xml_file_path)
    
    # Save as MIDI
    midi = score.write('midi', fp=midi_file_path)
    print(f"MIDI file saved as {midi_file_path}")



#--------------MIDI to String(Of notes)(Not Part Of Flow)-------------

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

def wav_to_midi_improved(input_wav, output_midi, params):
    # Load the audio file
    y, sr = librosa.load(input_wav)
    
    # Get total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=params['sample_rate'])

    # Use pYIN for improved pitch detection
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                 fmin=librosa.note_to_hz(params['min_note']), 
                                                 fmax=librosa.note_to_hz(params['max_note']),
                                                 frame_length=params['frame_length'],
                                                 hop_length=params['hop_length'])

    # Create a MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, params['tempo'])

    # Set the instrument to Acoustic Grand Piano
    midi.addProgramChange(track, 0, time, params['instrument'])

    # Process the pitch data
    last_pitch = None
    start_time = 0
    hop_time = params['hop_length'] / len(f0)

    for i, pitch in enumerate(f0):
        current_time = i * hop_time
        if voiced_flag[i] and pitch is not None:
            midi_pitch = round(librosa.hz_to_midi(pitch))
            if midi_pitch != last_pitch:
                if last_pitch is not None:
                    duration = current_time - start_time
                    if duration >= params['min_duration']:
                        velocity = min(127, max(1, int(voiced_probs[i] * 127)))
                        midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                start_time = current_time
                last_pitch = midi_pitch
        else:
            if last_pitch is not None:
                duration = current_time - start_time
                if duration >= params['min_duration']:
                    velocity = min(127, max(1, int(voiced_probs[i-1] * 127)))
                    midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                last_pitch = None

    # Add the last note if there is one
    if last_pitch is not None:
        duration = total_duration - start_time
        if duration >= params['min_duration']:
            velocity = min(127, max(1, int(voiced_probs[-1] * 127)))
            midi.addNote(track, 0, last_pitch, start_time, duration, velocity)

    # Save the MIDI file
    with open(output_midi, "wb") as output_file:
        midi.writeFile(output_file)






if __name__ == "__main__":
   # Adjustable parameters
    params = {
        'sample_rate': 44100,  # Audio sample rate
        'min_note': 'C2',      # Lowest note to detect
        'max_note': 'C7',      # Highest note to detect
        'frame_length': 2048,  # Frame size for pitch detection
        'hop_length': 512,     # Hop size for pitch detection
        'min_duration': 0.1,   # Minimum note duration in seconds
        'tempo': 120,          # MIDI tempo
        'instrument': 0        # MIDI instrument (0 = Acoustic Grand Piano)
    }

    # Use the function with the specified input and output paths
    wav_to_midi_improved('TestMelodies/M1.wav', 'TestMelodies/Inputs/ModelInputMIDI.mid', params)   
    
    
    
    # Example usage
    #convert_xml_to_midi('C://Users//user//Documents//GitHub//ChordGen//Example//as.xml', 'C://Users//user//Documents//GitHub//ChordGen//Example//test.mid')

    # # Converting Melody MIDI to Melody String Sequence
    # midi_file = "TestMelodies/Inputs/melody.mid"
    # note_string = midi_to_note_string(midi_file)
    # print(note_string)
