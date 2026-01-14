import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import display, Audio  # Explicit import
import IPython.display as ipd  # Alternative import
import os

plt.style.use('ggplot')
#%matplotlib inline
def load_and_visualize_audio(file_path, target_sr=16000):
    """Load audio file and display waveform and spectrogram"""
    # Load audio
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Plot spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set_title('Spectrogram')
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    # Create audio player
    display(ipd.Audio(audio, rate=sr))
    
    return audio, sr

# Example usage:
file_path =  r"C:\Users\Hamid Ali\Downloads\gym-ambience-v2-58673.mp3"  # Replace with your file path
audio, sr = load_and_visualize_audio(file_path)
def spectral_subtraction(noisy_audio, sr, n_fft=2048, hop_length=512, noise_frame_start=0, noise_frame_end=5):
    """Basic spectral subtraction noise reduction"""
    # Compute STFT
    stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    
    # Estimate noise from first few frames
    noise_mag = np.mean(magnitude[:, noise_frame_start:noise_frame_end], axis=1, keepdims=True)
    
    # Subtract noise estimate
    clean_mag = magnitude - noise_mag
    clean_mag = np.maximum(clean_mag, 0)  # Avoid negative values  
    # Reconstruct audio
    clean_stft = clean_mag * phase
    clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
    return clean_audio

def apply_noise_reduce(noisy_audio, sr, stationary=False, prop_decrease=1.0, n_fft=2048):
    """Apply noisereduce library's advanced noise reduction"""
    # Use first 100ms for noise profile
    noise_clip = noisy_audio[:int(0.1*sr)] 
    return nr.reduce_noise(
        y=noisy_audio,
        y_noise=noise_clip,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
        n_fft=n_fft
    )
def process_audio(file_path, output_dir="enhanced_results"):
    """Complete audio enhancement pipeline"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and visualize original audio
    print("Original Audio:")
    noisy_audio, sr = load_and_visualize_audio(file_path)
    # Apply noise reduction methods
    methods = {
        "Spectral Subtraction": spectral_subtraction(noisy_audio, sr),
        "NoiseReduce": apply_noise_reduce(noisy_audio, sr)
    }   
    # Process and save results
    for name, enhanced_audio in methods.items():
        # Save enhanced audio
        output_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.wav")
        sf.write(output_path, enhanced_audio, sr)
        # Visualize results
        print(f"\n{name} Enhanced Audio:")
        load_and_visualize_audio(output_path)   
    return methods
# Run the processing
enhanced_audios = process_audio(file_path)
def evaluate_enhancement(clean_audio, noisy_audio, enhanced_audio, sr):
    """Calculate quality metrics"""
    metrics = {}
    # Calculate SNR
    def calculate_snr(clean, noisy):
        noise = noisy - clean
        return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
    
    metrics["SNR_original"] = calculate_snr(clean_audio, noisy_audio)
    metrics["SNR_enhanced"] = calculate_snr(clean_audio, enhanced_audio)
    metrics["SNR_improvement"] = metrics["SNR_enhanced"] - metrics["SNR_original"]
    
    # Calculate PESQ (if same length)
    min_length = min(len(clean_audio), len(enhanced_audio))
    metrics["PESQ"] = pesq(sr, clean_audio[:min_length], enhanced_audio[:min_length], 'wb')
    
    return metrics

# Example usage (requires clean reference audio):
# clean_audio, _ = librosa.load("clean_reference.wav", sr=sr)
# metrics = evaluate_enhancement(clean_audio, noisy_audio, enhanced_audios["NoiseReduce"], sr)
# print(pd.DataFrame([metrics]))
