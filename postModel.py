from music21 import stream, chord, midi, converter, note, meter
#--------------------Combining MIDI-----------------------

def combine_midi_files(melody_file, chords_file, output_file):
    """ Combines melody and chord MIDI files into a single MIDI file. """
    # Load the melody and chords MIDI files
    melody_stream = converter.parse(melody_file)
    chords_stream = converter.parse(chords_file)

    # Ensure measures are added to the streams
    melody_stream = add_measures(melody_stream)
    chords_stream = add_measures(chords_stream)

    # Create a new stream for the combined output
    combined_stream = stream.Score()
    combined_stream.append(melody_stream)
    combined_stream.append(chords_stream)

    # Write the combined stream to a new MIDI file
    midi_file = midi.translate.streamToMidiFile(combined_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"Combined MIDI file '{output_file}' has been saved.")


def add_measures(midi_stream):
    """ Encloses notes in measures for proper MIDI structuring. """
    measures = stream.Part()
    measures.append(meter.TimeSignature('4/4'))
    for element in midi_stream.flat.notesAndRests:
        measures.append(element)
    measures.makeMeasures(inPlace=True)
    return measures


#------------------ String to MIDI---------------------


from music21 import note, chord

def chord_notes(root, quality):
    """ Generate the correct notes for a given chord root and quality with default velocity. """
    notes = []
    velocity = 64  # A sensible default velocity for audible notes
    # Define each chord type with appropriate intervals and ensure the velocity is set
    if quality == 'major':
        notes = [note.Note(root+'3', volume=velocity), 
                 note.Note(root+'5', volume=velocity), 
                 note.Note(root+'8', volume=velocity)]
    elif quality == 'minor':
        notes = [note.Note(root+'b3', volume=velocity), 
                 note.Note(root+'5', volume=velocity), 
                 note.Note(root+'8', volume=velocity)]
    elif quality == 'dominant':
        notes = [note.Note(root+'3', volume=velocity), 
                 note.Note(root+'5', volume=velocity), 
                 note.Note(root+'b7', volume=velocity)]
    elif quality == 'major-seventh':
        notes = [note.Note(root+'3', volume=velocity), 
                 note.Note(root+'5', volume=velocity), 
                 note.Note(root+'7', volume=velocity)]
    elif quality == 'minor-seventh':
        notes = [note.Note(root+'b3', volume=velocity), 
                 note.Note(root+'5', volume=velocity), 
                 note.Note(root+'b7', volume=velocity)]
    return chord.Chord(notes)




from music21 import stream, midi, converter, meter

def change_chords_to_midi(chord_string, timings, melody_file, output_file):
    melody_stream = converter.parse(melody_file)
    total_duration = melody_stream.duration.quarterLength  # Get total length of the melody in quarter lengths
    
    chords_part = stream.Part()
    time_signature = melody_stream.recurse().getElementsByClass(meter.TimeSignature)[0]
    chords_part.append(time_signature)  # Copy time signature from melody

    chord_entries = chord_string.split(',')
    for idx, chord_entry in enumerate(chord_entries):
        root, quality = chord_entry.split('/')
        new_chord = chord_notes(root, quality)
        new_chord.duration.quarterLength = 4  # Adjust duration as needed
        scaled_timing = (timings[idx] / max(timings)) * total_duration
        new_chord.offset = scaled_timing  # Set offset based on scaled timing
        chords_part.append(new_chord)

    combined_stream = stream.Score()
    combined_stream.append(melody_stream.parts[0])  # Assuming melody is single-part
    combined_stream.append(chords_part)

    midi_file = midi.translate.streamToMidiFile(combined_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"Combined MIDI file '{output_file}' has been saved.")


#-------------------Melody + Chord String + BMP to MIDI----------Retrying above functions
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
    print(ticks_per_beat)
    tempo = mido.bpm2tempo(bpm)
    track.append(MetaMessage('set_tempo', tempo=tempo))

    time_since_last_event = 0  # Initialize time offset

    for melody, chord in zip(melody_notes, chord_progressions):
        chord_notes = chord_to_notes(chord)

        # Process Chord Notes
        for note in chord_notes:
            track.append(Message('note_on', note=note, velocity=64, time=time_since_last_event))
            time_since_last_event = 0  # Reset time for subsequent events in this step

        for note in chord_notes:
            track.append(Message('note_off', note=note, velocity=64, time=ticks_per_beat))
            time_since_last_event = 0  # Reset time for subsequent events in this step

        # Process Melody Note
        if melody != "PAD":
            melody_note = note_to_midi(melody)
            if melody_note is not None:
                track.append(Message('note_on', note=melody_note, velocity=64, time=time_since_last_event))
                track.append(Message('note_off', note=melody_note, velocity=64, time=ticks_per_beat + 1000))
                time_since_last_event = 0  # Reset time for the next event

    mid.save('SungMelodies/Outputs/output_midi.mid')

# Example usage
if __name__ == "__main__":
    melody_string = "C0/6,C1/6,C0/6,B0/5,A0/5,G0/5,F1/5,D0/5,F1/5,E0/5,B0/5,F0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,D0/5,D0/5,F1/5,A0/5,C1/6,E0/6,D0/6,D0/5,C1/5,D0/5,E0/6,E0/6,D0/6,B0/5,A0/5,F1/5"
    chord_string = "PAD,E0/dominant,E0/dominant,E0/minor-seventh,E0/minor-seventh,B0/minor-seventh,E0/minor-seventh,B0/minor-seventh,B0/minor-seventh,B0/minor-seventh,E0/dominant,E0/dominant,B0/minor-seventh,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,D0/dominant,D0/dominant,G0/major-seventh,G0/major-seventh,G0/major-seventh,E0/dominant,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A1/diminished,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A1/diminished,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A1/diminished,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A1/diminished"
    bpm = 120
    create_midi(melody_string, chord_string, bpm)



    # chord_string = "E0/dominant,E0/dominant,E0/minor-seventh,E0/minor-seventh,B0/minor-seventh,E0/minor-seventh,B0/minor-seventh,B0/minor-seventh,B0/minor-seventh,E0/dominant,E0/dominant,B0/minor-seventh,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,D0/dominant,D0/dominant,G0/major-seventh,G0/major-seventh,G0/major-seventh,E0/dominant,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A1/diminished,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A1/diminished,A0/minor-seventh,A0/minor-seventh,D0/dominant,A1/diminished,A1/diminished,A1/diminished,A0/minor-seventh,A0/minor-seventh,A0/minor-seventh,A1/diminished"
    # melody_file_path = "SungMelodies/Inputs/melody.mid"
    # output_file_path = "SungMelodies/Outputs/chords.mid"
    # change_chords_to_midi(chord_string, timings, melody_file_path, output_file_path)


    # # Combining the Melody and Chord MIDIs
    # chords_file = "SungMelodies/Outputs/chords.mid"
    # output_file = "SungMelodies/Outputs/combined_output.mid"
    # combine_midi_files(melody_file, chords_file, output_file)
