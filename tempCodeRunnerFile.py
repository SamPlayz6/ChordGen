import librosa
import numpy as np
from midiutil import MIDIFile

def wav_to_midi_improved(input_wav, output_midi, min_duration=0.1):
    # Load the audio file
    y, sr = librosa.load(input_wav)
    
    # Get total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Use pYIN for improved pitch detection
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'))

    # Create a MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, 120)

    # Set the instrument to Acoustic Grand Piano
    program = 0
    midi.addProgramChange(track, 0, time, program)

    # Process the pitch data
    last_pitch = None
    start_time = 0
    hop_time = total_duration / len(f0)

    for i, pitch in enumerate(f0):
        current_time = i * hop_time
        if voiced_flag[i] and pitch is not None:
            midi_pitch = round(librosa.hz_to_midi(pitch))
            if midi_pitch != last_pitch:
                if last_pitch is not None:
                    duration = current_time - start_time
                    if duration >= min_duration:
                        velocity = min(127, max(1, int(voiced_probs[i] * 127)))
                        midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                start_time = current_time
                last_pitch = midi_pitch
        else:
            if last_pitch is not None:
                duration = current_time - start_time
                if duration >= min_duration:
                    velocity = min(127, max(1, int(voiced_probs[i-1] * 127)))
                    midi.addNote(track, 0, last_pitch, start_time, duration, velocity)
                last_pitch = None

    # Add the last note if there is one
    if last_pitch is not None:
        duration = total_duration - start_time
        if duration >= min_duration:
            velocity = min(127, max(1, int(voiced_probs[-1] * 127)))
            midi.addNote(track, 0, last_pitch, start_time, duration, velocity)

    # Save the MIDI file
    with open(output_midi, "wb") as output_file:
        midi.writeFile(output_file)