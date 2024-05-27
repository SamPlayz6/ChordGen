from music21 import converter

def convert_xml_to_midi(xml_file_path, midi_file_path):
    # Load the MusicXML file
    score = converter.parse(xml_file_path)
    
    # Save as MIDI
    midi = score.write('midi', fp=midi_file_path)
    print(f"MIDI file saved as {midi_file_path}")

# Example usage
convert_xml_to_midi('C://Users//user//Documents//GitHub//ChordGen//Example//as.xml', 'C://Users//user//Documents//GitHub//ChordGen//Example//test.mid')