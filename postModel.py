from music21 import stream, chord, midi, converter, note, meter

def chord_notes(root, quality):
    """ Generate the correct notes for a given chord root and quality. """
    notes = []
    if quality == 'major':
        notes = [root + '3', root + '5', root + '8']  # Simple major triad
    elif quality == 'minor':
        notes = [root + '3', root + 'b5', root + '8']  # Minor triad
    elif quality == 'dominant':
        notes = [root + '3', root + '5', root + 'b7']  # Dominant seventh
    elif quality == 'major-seventh':
        notes = [root + '3', root + '5', root + '7']  # Major seventh
    elif quality == 'minor-seventh':
        notes = [root + '3', root + 'b5', root + 'b7']  # Minor seventh
    # Add more conditions for other types like diminished, augmented if necessary
    return notes

def add_measures(midi_stream):
    measures = stream.Part()
    measures.append(meter.TimeSignature('4/4'))
    for element in midi_stream.flat.notesAndRests:
        measures.append(element)
    measures.makeMeasures(inPlace=True)
    measures.makeTies(inPlace=True)
    return measures

def chords_to_midi(chord_string, timings, output_file='SungMelodies/Outputs/chords.mid'):
    chord_list = chord_string.split(',')
    chords = []
    for i, chord_name in enumerate(chord_list):
        root, quality = chord_name.split('/')
        notes = chord_notes(root, quality)
        new_chord = chord.Chord(notes)
        new_chord.quarterLength = 1  # Default duration, adjust as needed
        chords.append((new_chord, timings[i]))

    chords_stream = stream.Stream()
    for ch, start_time in chords:
        ch.offset = start_time
        chords_stream.append(ch)

    # Ensure measures are added
    chords_stream = add_measures(chords_stream)

    midi_file = midi.translate.music21ObjectToMidiFile(chords_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"MIDI file '{output_file}' has been saved.")

def combine_midi_files(melody_file, chords_file, output_file):
    # Load the melody and chords MIDI files
    melody_stream = converter.parse(melody_file)
    chords_stream = converter.parse(chords_file)

    # Ensure measures are added to the streams
    melody_stream = add_measures(melody_stream)
    chords_stream = add_measures(chords_stream)

    # Create a new stream for the combined output
    combined_stream = stream.Score()

    # Add the melody part to the combined stream
    combined_stream.append(melody_stream)

    # Add the chords part to the combined stream
    chords_part = stream.Part()
    for element in chords_stream.flat.notes:
        if isinstance(element, chord.Chord):
            chords_part.append(element)
        elif isinstance(element, note.Note):
            chords_part.append(element)

    combined_stream.append(chords_part)

    # Write the combined stream to a new MIDI file
    midi_file = midi.translate.music21ObjectToMidiFile(combined_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"Combined MIDI file '{output_file}' has been saved.")

if __name__ == "__main__":
    # # Example usage
    # chord_string = "C4/major,G4/minor,C4/major-seventh,A4/minor-seventh"
    # timings = [0, 2, 4, 6]  # Start times for each chord
    # chords_to_midi(chord_string, timings)

    # Combining the Melody and Chord MIDIs
    melody_file = "SungMelodies/Inputs/melody.mid"
    chords_file = "SungMelodies/Outputs/chords.mid"
    output_file = "SungMelodies/Outputs/combined_output.mid"
    combine_midi_files(melody_file, chords_file, output_file)