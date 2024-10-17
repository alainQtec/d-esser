import numpy as np
import librosa
import soundfile as sf

def deesser(audio_file, frequency, threshold):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # Apply a low-pass filter to emphasize sibilant frequencies
    y_lowpass = librosa.effects.preemphasis(y)
    
    # Calculate the amplitude envelope
    envelope = librosa.onset.onset_strength(y=y_lowpass, sr=sr, aggregate=np.median)
    
    # Interpolate the envelope to match the audio data dimensions
    envelope_interp = np.interp(np.arange(len(y)), np.linspace(0, len(y), len(envelope)), envelope)
    
    # Apply selective compression based on the envelope and threshold
    gain = np.where(envelope_interp > threshold, 1.0, 0.0)
    y_deessed = y * gain
    
    # Save the processed audio to a new file
    sf.write("audio_deessed.wav", y_deessed, sr)

# Example usage
audio_file = "audio_input.wav"

# Choose the desired cutoff frequency
frequency = float(input("Choose the desired cutoff frequency: "))  # Cutoff frequency for sibilance

threshold = float(input("Choose the desired threshold: "))  # Threshold for compression activation
threshold = np.float32(threshold)  # Convert to float32 data type

deesser(audio_file, frequency, threshold)
