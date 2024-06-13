from music21 import stream, chord, midi, converter, note, meter

def chord_notes(root, quality):
    """ Generate the correct notes for a given chord root and quality. """
    notes = []
    if quality == 'major':
        notes = [root + '3', root + '5', root + '8']  # Simple major triad
    elif quality == 'minor':
        notes = [root + 'b3', root + '5', root + '8']  # Minor triad
    elif quality == 'dominant':
        notes = [root + '3', root + '5', root + 'b7']  # Dominant seventh
    elif quality == 'major-seventh':
        notes = [root + '3', root + '5', root + '7']  # Major seventh
    elif quality == 'minor-seventh':
        notes = [root + 'b3', root + '5', root + 'b7']  # Minor seventh
    return notes

def add_measures(midi_stream):
    """ Encloses notes in measures for proper MIDI structuring. """
    measures = stream.Part()
    measures.append(meter.TimeSignature('4/4'))
    for element in midi_stream.flat.notesAndRests:
        measures.append(element)
    measures.makeMeasures(inPlace=True)
    return measures

def chords_to_midi(chord_string, timings, output_file='SungMelodies/Outputs/chords.mid'):
    """ Converts a string of chords and their timings into a MIDI file. """
    chords_stream = stream.Stream()
    for chord_info, time in zip(chord_string.split(','), timings):
        root, quality = chord_info.split('/')
        chord_obj = chord.Chord(chord_notes(root, quality))
        chord_obj.quarterLength = 1  # Set chord duration, adjust as needed
        chord_obj.offset = time  # Set start time for each chord
        chords_stream.append(chord_obj)

    # Wrap chords in measures if needed
    chords_stream = add_measures(chords_stream)

    # Convert stream to MIDI and save
    midi_file = midi.translate.streamToMidiFile(chords_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"MIDI file '{output_file}' has been saved.")

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

if __name__ == "__main__":
    # Example usage
    chord_string = "C4/major,G4/minor,C4/major-seventh,A4/minor-seventh"
    timings = [0, 2, 4, 6]  # Example timings for each chord
    melody_file = "SungMelodies/Inputs/melody.mid"
    chords_to_midi(chord_string, timings)

    # # Combining the Melody and Chord MIDIs
    # chords_file = "SungMelodies/Outputs/chords.mid"
    # output_file = "SungMelodies/Outputs/combined_output.mid"
    # combine_midi_files(melody_file, chords_file, output_file)
