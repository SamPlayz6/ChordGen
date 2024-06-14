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


#------------------ String to MIDI---------------------------


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
    """ Ensures every note or rest in the stream is within a measure structure. """
    if not midi_stream.getElementsByClass('Measure'):
        # If no measures, wrap all elements in one measure
        new_stream = stream.Measure()
        for el in midi_stream.notesAndRests:
            new_stream.append(el)
        midi_stream = stream.Part()
        midi_stream.append(new_stream)
    midi_stream.makeMeasures(inPlace=True)  # This will now work because everything is wrapped in a Measure
    return midi_stream

def add_chords_to_midi(chord_string, timings, melody_file, output_file):
    melody_stream = converter.parse(melody_file)
    # Ensure that melody_stream is properly measured
    melody_measures = add_measures(melody_stream)

    chords_part = stream.Part()
    chord_entries = chord_string.split(',')
    for idx, chord_entry in enumerate(chord_entries):
        root, quality = chord_entry.split('/')
        chord_notes_list = chord_notes(root, quality)
        new_chord = chord.Chord(chord_notes_list, quarterLength=4)
        new_chord.offset = timings[idx]  # Directly use timings without scaling
        chords_part.append(new_chord)

    chords_measured = add_measures(chords_part)
    combined_stream = stream.Score()
    combined_stream.append(melody_measures)
    combined_stream.append(chords_measured)

    # Now ensure the combined_stream does not need expandRepeats or contains valid measures
    midi_file = midi.translate.streamToMidiFile(combined_stream)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    print(f"Combined MIDI file '{output_file}' has been saved.")

if __name__ == "__main__":
    chord_string = "C4/major,G4/minor,C4/major-seventh,A4/minor-seventh"
    timings = [0, 2, 4, 6]
    melody_file = "SungMelodies/Inputs/melody.mid"
    output_file = "SungMelodies/Outputs/combined_output.mid"
    add_chords_to_midi(chord_string, timings, melody_file, output_file)

    # # Combining the Melody and Chord MIDIs
    # chords_file = "SungMelodies/Outputs/chords.mid"
    # output_file = "SungMelodies/Outputs/combined_output.mid"
    # combine_midi_files(melody_file, chords_file, output_file)
