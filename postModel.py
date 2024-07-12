import argparse
#-------------------Convert Melody to MIDI(Not Part Of Flow)-----------
def create_melody_midi(melody_string, bpm):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set the tempo
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

    ticks_per_beat = mid.ticks_per_beat
    melody_notes = melody_string.split(',')

    for i, note_str in enumerate(melody_notes):
        note_number = note_to_midi(note_str)
        velocity = 64  # standard velocity
        time = 0
        
        # Note on
        track.append(Message('note_on', note=note_number, velocity=velocity, time=time))
        # Note off, here we assume each note lasts for one beat
        track.append(Message('note_off', note=note_number, velocity=velocity, time=ticks_per_beat))

    # Save the MIDI file
    mid.save('Input/Output/inference_melody_midi.mid')



#-------------------Melody + Chord String + BMP to MIDI-------------------
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Convert note string to MIDI note number
def note_to_midi(note_str):
    """ Convert note string with format 'NoteSharpOrNatural/Octave' to MIDI note number. """
    if note_str in {"PAD", "UNK"}:
        return None  # Return None for PAD or UNK

    note_part, octave_str = note_str.split('/')
    note = note_part[0]  # Note letter (e.g., 'C')
    sharp_or_natural = note_part[1]  # '1' for sharp, '0' for natural
    octave = int(octave_str)  # Octave number

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if sharp_or_natural == '1':
        note_name = note + '#'
    else:
        note_name = note

    base_note_index = notes.index(note_name)

    # MIDI notes start at 0 for C-1 (MIDI standard), so middle C (C4) is 60
    return 12 * (octave + 1) + base_note_index

# Process chords to get individual notes
def chord_to_notes(chord):
    if chord == "PAD":
        return []  # Return empty list for PAD

    root, quality = chord.split('/')
    root_note = note_to_midi(root + "/4")

    if root_note is None:
        return []

    if 'minor' in quality:
        return [root_note, root_note + 3, root_note + 7]
    elif 'dominant' in quality:
        return [root_note, root_note + 4, root_note + 7, root_note + 10]
    elif 'major-seventh' in quality:
        return [root_note, root_note + 4, root_note + 7, root_note + 11]
    elif 'diminished' in quality:
        return [root_note, root_note + 3, root_note + 6]
    else:
        return [root_note]  # Default to a single note if no quality matches

def create_midi(melody_string, chord_string, bpm):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    melody_notes = melody_string.split(',')
    chord_progressions = chord_string.split(',')

    ticks_per_beat = mid.ticks_per_beat
    # ticks_per_step = int(ticks_per_beat * (60 / bpm))

    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

    current_chord_notes = None
    time_since_last_note = 0

    for melody_note, chord in zip(melody_notes, chord_progressions):
        new_chord_notes = chord_to_notes(chord)
        melody_note_midi = note_to_midi(melody_note)

        # Process Chord Changes
        if new_chord_notes != current_chord_notes:
            if current_chord_notes is not None:
                for note in current_chord_notes:
                    track.append(Message('note_off', note=note, velocity=64, time=0))
            for note in new_chord_notes:
                track.append(Message('note_on', note=note, velocity=64, time=time_since_last_note))
            current_chord_notes = new_chord_notes
            time_since_last_note = 0

        # Process Melody Note
        if melody_note_midi is not None:
            track.append(Message('note_on', note=melody_note_midi, velocity=64, time=time_since_last_note))
            track.append(Message('note_off', note=melody_note_midi, velocity=64, time=ticks_per_beat))
            time_since_last_note = 0

    # Ensure all notes are turned off at the end
    if current_chord_notes:
        for note in current_chord_notes:
            track.append(Message('note_off', note=note, velocity=64, time=0))

    mid.save('Input/Output/inference_song_midi.mid')


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input Strings for MIDI reconstruction')
    parser.add_argument('melody_input', nargs='?', help='melody String input')
    parser.add_argument('chord_output', nargs='?', help='Chord String output')


    args = parser.parse_args()

    melody_string = args.melody_input
    chord_string = args.chord_output
    bpm = 110

    create_midi(melody_string, chord_string, bpm)
    create_melody_midi(melody_string,bpm)
