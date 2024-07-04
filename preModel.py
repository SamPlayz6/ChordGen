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

def wav_to_midi(input_wav, output_midi, sampling_rate=100, min_duration=0.1, noise_threshold=0.1):
    # Load the audio file
    y, sr = librosa.load(input_wav)

    # Extract pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Create a MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, 120)

    # Set the instrument to Acoustic Grand Piano
    program = 0
    midi.addProgramChange(track, 0, time, program)

    # Calculate the hop length used by librosa.piptrack
    hop_length = len(y) // (pitches.shape[1] - 1)

    last_pitch = None
    start_time = 0
    total_duration = librosa.get_duration(y=y, sr=sr)

    for frame in range(pitches.shape[1]):
        # Calculate the corresponding time in the original audio
        current_time = frame * hop_length / sr

        # Get the pitch with the highest magnitude
        index = magnitudes[:, frame].argmax()
        pitch = pitches[index, frame]
        magnitude = magnitudes[index, frame]

        if pitch > 0 and magnitude > noise_threshold:  # Check if the pitch is above the noise threshold
            midi_pitch = round(librosa.hz_to_midi(pitch))

            if midi_pitch != last_pitch:
                if last_pitch is not None:
                    # Add the previous note to the MIDI file
                    duration = current_time - start_time
                    if duration >= min_duration:
                        midi.addNote(track, 0, last_pitch, start_time, duration, 100)

                # Start a new note
                start_time = current_time
                last_pitch = midi_pitch
        else:
            # No clear pitch or below noise threshold, end the current note if there is one
            if last_pitch is not None:
                duration = current_time - start_time
                if duration >= min_duration:
                    midi.addNote(track, 0, last_pitch, start_time, duration, 100)
                last_pitch = None

    # Add the last note if there is one
    if last_pitch is not None:
        duration = total_duration - start_time
        if duration >= min_duration:
            midi.addNote(track, 0, last_pitch, start_time, duration, 100)

    # Save the MIDI file
    with open(output_midi, "wb") as output_file:
        midi.writeFile(output_file)




if __name__ == "__main__":
    # Use the function with the specified input and output paths
    wav_to_midi('TestMelodies/M1.wav', 'TestMelodies/Inputs/ModelInputMIDI.mid', noise_threshold=0.9)

    # Example usage
    #convert_xml_to_midi('C://Users//user//Documents//GitHub//ChordGen//Example//as.xml', 'C://Users//user//Documents//GitHub//ChordGen//Example//test.mid')

    # # Converting Melody MIDI to Melody String Sequence
    # midi_file = "TestMelodies/Inputs/melody.mid"
    # note_string = midi_to_note_string(midi_file)
    # print(note_string)
