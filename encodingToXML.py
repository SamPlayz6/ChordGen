import xml.etree.ElementTree as ET

def write_xml_file(melodies, chords, output_file):
    # Create the root element
    root = ET.Element("root")
    
    # Assume a single part for simplicity
    part = ET.SubElement(root, "part")
    
    # Track measures
    measure_number = 1
    measure = ET.SubElement(part, "measure", number=str(measure_number))
    
    # Process melodies and chords assuming they align by measure
    for melody_note, chord_symbol in zip(melodies, chords):
        # When the melody changes measure (optional, if your data structure requires)
        # if melody_note indicates a new measure:
        #    measure_number += 1
        #    measure = ET.SubElement(part, "measure", number=str(measure_number))
        
        # Add melody note
        note = ET.SubElement(measure, "note")
        pitch = ET.SubElement(note, "pitch")
        
        step, octave = melody_note.split('/')  # assuming melody_note is like "C#4/5"
        alter = '0'
        if '#' in step or '-' in step:
            alter = '1' if '#' in step else '-1'
            step = step.strip('#-')
        
        ET.SubElement(pitch, "step").text = step
        ET.SubElement(pitch, "octave").text = octave
        if alter != '0':  # Only add alter if necessary
            ET.SubElement(pitch, "alter").text = alter
        
        # Add chord
        harmony = ET.SubElement(measure, "harmony")
        root_elem = ET.SubElement(harmony, "root")
        chord_step, kind = chord_symbol.split('/')  # assuming chord_symbol is like "G1/major"
        chord_alter = '0'
        if '#' in chord_step or '-' in chord_step:
            chord_alter = '1' if '#' in chord_step else '-1'
            chord_step = chord_step.strip('#-')
        
        ET.SubElement(root_elem, "root-step").text = chord_step
        if chord_alter != '0':
            ET.SubElement(root_elem, "root-alter").text = chord_alter
        ET.SubElement(harmony, "kind").text = kind

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"XML file '{output_file}' written successfully.")

# Example usage:
# melodies = ["C#4/5", "D4/5", "E4/5"]  # These should be the outputs from your model, representing notes
# chords = ["G1/major", "A1/minor", "B1/7"]  # These should also be outputs from your model, representing chords
# write_xml_file(melodies, chords, "output_music.xml"