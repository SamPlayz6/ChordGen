import mido
from mido import Message, MidiFile, MidiTrack

def create_midi(chords, tempo=500000):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    track.append(Message('program_change', program=12, time=0))
    delta = 480  # Duration of each note
    for chord in chords:
        note_number = chord_name_to_midi_number(chord)  # Convert chord to MIDI note number
        track.append(Message('note_on', note=note_number, velocity=64, time=0))
        track.append(Message('note_off', note=note_number, velocity=64, time=delta))
    
    mid.save('Example/Outputs/output.mid')

def chord_name_to_midi_number(chord):
    # Conversion logic from chord name to MIDI note number
    return 60  # Example: Return middle C for simplicity

# Example usage:
predicted_chords = ['C', 'E', 'G']  # Example predicted chords
create_midi(predicted_chords)
