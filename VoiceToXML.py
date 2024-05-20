import librosa
import music21

def audio_to_pitches_and_durations(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Simple pitch extraction logic
    pitch_values = []
    for pitch_col in pitches.T:
        index = pitch_col.argmax()
        pitch_values.append(pitches[index, :].mean())
    # Convert frequency to musical notes, this is a placeholder
    notes = [music21.pitch.Pitch(p).nameWithOctave for p in pitch_values if p > 0]
    durations = [1.0] * len(notes)  # simple constant duration, placeholder
    return notes, durations

def create_music_xml(notes, durations, file_name):
    # Create a music21 stream
    s = music21.stream.Score()
    part = music21.stream.Part()
    for n, d in zip(notes, durations):
        note = music21.note.Note(n)
        note.duration = music21.duration.Duration(d)
        part.append(note)
    s.append(part)
    
    # Setting up the metadata as per your example
    s.metadata = music21.metadata.Metadata(title='Generated Song', composer='Composer')
    
    # Write to MusicXML
    s.write('musicxml', fp=f"{file_name}.xml")

# Example usage
audio_file = 'C://Users//user//Documents//GitHub//ChordGen//SungMelodies//M1.wav'
notes, durations = audio_to_pitches_and_durations(audio_file)
create_music_xml(notes, durations, 'C://Users//user//Documents//GitHub//ChordGen//SungMelodies//Outputs//example')