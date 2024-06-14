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

def add_chords_to_midi(chord_string, timings, melody_file, output_file):
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

# Example usage
if __name__ == "__main__":
    chord_string = "C4/major,G4/minor,C4/major-seventh,A4/minor-seventh"
    timings = [0, 2, 4, 6]  # Example timings
    melody_file_path = "SungMelodies/Inputs/melody.mid"
    output_file_path = "SungMelodies/Outputs/chords.mid"
    add_chords_to_midi(chord_string, timings, melody_file_path, output_file_path)



    # # Combining the Melody and Chord MIDIs
    # chords_file = "SungMelodies/Outputs/chords.mid"
    # output_file = "SungMelodies/Outputs/combined_output.mid"
    # combine_midi_files(melody_file, chords_file, output_file)
