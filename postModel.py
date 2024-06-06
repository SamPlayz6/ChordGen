from music21 import stream, chord, duration, midi, converter

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

def chords_to_midi(chord_string, timings, output_file='SungMelodies/Outputs/chords.mid'):
    # Split the input string into a list of chords
    chord_list = chord_string.split(',')

    # Create a Stream to hold the chords
    s = stream.Stream()

    # Process each chord with corresponding timing
    for i, ch in enumerate(chord_list):
        # Extract the root and quality
        parts = ch.split('/')
        root = parts[0]
        quality = parts[1] if len(parts) > 1 else 'major'  # Default to major if not specified

        # Create a chord object based on the root and quality
        chord_notes_list = chord_notes(root, quality)
        c = chord.Chord(chord_notes_list)
        c.duration = duration.Duration(1.0)  # Default duration, adjust as needed

        # Set the start time for the chord
        c.offset = timings[i]

        # Add the chord to the Stream
        s.append(c)

    # Write the stream to a MIDI file
    mf = midi.translate.streamToMidiFile(s)
    mf.open(output_file, 'wb')
    mf.write()
    mf.close()
    print(f"MIDI file '{output_file}' has been saved.")


def combine_midi_files(melody_file, chords_file, output_file):
    # Load the melody and chords MIDI files
    melody_stream = converter.parse(melody_file)
    chords_stream = converter.parse(chords_file)

    # Create a new stream to hold the combined MIDI
    combined_stream = stream.Stream()

    # Add melody and chords to the combined stream
    for element in melody_stream.flat.notes:
        combined_stream.append(element)
    for element in chords_stream.flat.chords:
        combined_stream.append(element)

    # Write the combined stream to a new MIDI file
    midi_file = midi.translate.music21ObjectToMidiFile(combined_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"Combined MIDI file '{output_file}' has been saved.")



if __name__ == "__main__":

    # Converting Chord names and timings to MIDI (Inference - Chords)
    chord_string = "C4/major,G4/minor,C4/major-seventh,A4/minor-seventh"
    timings = [0, 2, 4, 6]  # Start times for each chord
    chords_to_midi(chord_string, timings)


    # Combining the Melody and Chord MIDIs
    melody_file = "SungMelodies/Inputs/melody.mid"
    chords_file = "SungMelodies/Outputs/chords.mid"
    output_file = "SungMelodies/Outputs/combined_output.mid"
    combine_midi_files(melody_file, chords_file, output_file)


