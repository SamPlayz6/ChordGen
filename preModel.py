from music21 import converter
from mido import MidiFile

#--------------------XML to MIDI(Not Part Of Flow)-------------------

def convert_xml_to_midi(xml_file_path, midi_file_path):
    # Load the MusicXML file
    score = converter.parse(xml_file_path)
    
    # Save as MIDI
    midi = score.write('midi', fp=midi_file_path)
    print(f"MIDI file saved as {midi_file_path}")



#--------------MIDI to String(Of notes)(Not Part Of Flow)-------------

def midi_to_note_string(midi_file, ticks_per_beat=480):
    midi = MidiFile(midi_file)
    note_events = []
    current_time = 0
    note_string = []

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append((msg.note, current_time))
            if not msg.is_meta:
                current_time += msg.time

    # Convert note numbers to note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for note, time in note_events:
        note_name = note_names[note % 12]
        octave = note // 12 - 1  # MIDI note 60 is middle C (C4), which corresponds to note_name[0] in octave 4
        note_string.append(f"{note_name}{octave}/{time//ticks_per_beat}")

    return ",".join(note_string)


#-----------------AudioToMIDI(Or StringOfNotes)------------





if __name__ == "__main__":
    # Example usage
    #convert_xml_to_midi('C://Users//user//Documents//GitHub//ChordGen//Example//as.xml', 'C://Users//user//Documents//GitHub//ChordGen//Example//test.mid')


    # Converting Melody MIDI to Melody String Sequence
    midi_file = "TestMelodies/Inputs/melody.mid"
    note_string = midi_to_note_string(midi_file)
    print(note_string)
