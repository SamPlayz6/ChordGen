import librosa
import aubio
import numpy as np

def wav_to_midi_notes(filename, buffer_size=2048, hop_size=512):
    # Load the audio file
    data, sample_rate = librosa.load(filename, sr=None)

    # Setup for pitch detection
    p_detect = aubio.pitch("default", buffer_size, hop_size, sample_rate)
    p_detect.set_unit("Hz")
    p_detect.set_silence(-40)

    pitches = []
    confidences = []

    print(pitches)

    # Buffer processing
    for i in range(0, len(data), hop_size):
        samples = data[i:i+hop_size]
        if len(samples) < hop_size:
            samples = np.pad(samples, (0, hop_size - len(samples)), 'constant', constant_values=(0.0, 0.0))
        pitch = p_detect(samples)[0]
        confidence = p_detect.get_confidence()
        if confidence > 0.8:  # This threshold can be adjusted
            pitches.append(pitch)
            confidences.append(confidence)

    # Convert frequency to musical notes
    notes = [librosa.hz_to_note(pitch) for pitch in pitches if pitch > 0]
    return notes

def notes_to_input_sequence(notes):
    # Here you would convert notes to your desired format, e.g., "A0/5"
    # Placeholder: direct conversion, adapt this based on your specific needs
    input_sequence = ','.join(notes)
    return input_sequence

def process_wav_file(filename):
    notes = wav_to_midi_notes(filename)
    input_sequence = notes_to_input_sequence(notes)
    return input_sequence

# Usage
if __name__ == "__main__":
    filename = "SungMelodies/M1.wav"
    input_sequence = process_wav_file(filename)
    print("Generated Input Sequence:", input_sequence)