import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
import os
from io import BytesIO
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Audio Denoiser",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stSelectbox>div>div>div>div {
        background-color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        margin-top: 20px;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

def load_audio(file_path, target_sr=16000):
    """Load audio file with librosa"""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def plot_audio(audio, sr, title):
    """Create waveform and spectrogram plots"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Waveform plot
    librosa.display.waveshow(audio, sr=sr, ax=ax[0], color='#1f77b4')
    ax[0].set_title(f'{title} - Waveform', fontsize=12)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Spectrogram plot
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax[1], cmap='viridis')
    ax[1].set_title(f'{title} - Spectrogram', fontsize=12)
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def spectral_subtraction(noisy_audio, sr, n_fft=2048, hop_length=512, noise_frame_start=0, noise_frame_end=5):
    """Basic spectral subtraction noise reduction"""
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

def apply_noisereduce(noisy_audio, sr, stationary=False, prop_decrease=1.0, n_fft=2048):
    """Apply noisereduce library's advanced noise reduction"""
    noise_clip = noisy_audio[:int(0.1*sr)]  # Use first 100ms for noise profile
    return nr.reduce_noise(
        y=noisy_audio,
        y_noise=noise_clip,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
        n_fft=n_fft
    )

def save_audio(audio, sr, filename):
    """Save audio to bytes for download"""
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer

def main():
    st.title("🎧 Audio Denoiser")
    st.markdown("Upload an audio file to reduce background noise using different algorithms.")
    
    # Sidebar for file upload and parameters
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])
        
        if uploaded_file:
            st.markdown("---")
            st.subheader("Denoising Parameters")
            
            method = st.selectbox(
                "Denoising Method",
                ["Spectral Subtraction", "NoiseReduce"],
                index=1
            )
            
            if method == "Spectral Subtraction":
                n_fft = st.slider("FFT Size", 256, 4096, 2048, step=256)
                noise_duration = st.slider("Noise Estimation Duration (seconds)", 0.1, 5.0, 0.5, step=0.1)
            else:
                stationary = st.checkbox("Stationary Noise", value=False)
                prop_decrease = st.slider("Noise Reduction Amount", 0.1, 1.0, 0.8, step=0.1)
                n_fft = st.slider("FFT Size", 256, 4096, 2048, step=256)
    
    if uploaded_file:
        # Process the audio file
        try:
            # Load and display original audio
            st.subheader("Original Audio")
            audio, sr = load_audio(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.audio(uploaded_file, format='audio/wav')
            
            with col2:
                fig = plot_audio(audio, sr, "Original")
                st.pyplot(fig)
                plt.close()
            
            # Apply denoising
            st.subheader("Denoised Audio")
            
            if method == "Spectral Subtraction":
                noise_frames = int(noise_duration * sr / (n_fft / 4))  # Approximate frames
                enhanced_audio = spectral_subtraction(
                    audio, sr, 
                    n_fft=n_fft,
                    noise_frame_end=noise_frames
                )
            else:
                enhanced_audio = apply_noisereduce(
                    audio, sr,
                    stationary=stationary,
                    prop_decrease=prop_decrease,
                    n_fft=n_fft
                )
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                audio_buffer = save_audio(enhanced_audio, sr, "enhanced_audio.wav")
                st.audio(audio_buffer, format='audio/wav')
                
                # Download button
                st.download_button(
                    label="Download Enhanced Audio",
                    data=audio_buffer,
                    file_name="enhanced_audio.wav",
                    mime="audio/wav"
                )
            
            with col2:
                fig = plot_audio(enhanced_audio, sr, "Enhanced")
                st.pyplot(fig)
                plt.close()
            
            # Comparison section
            st.subheader("Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Audio**")
                st.audio(uploaded_file, format='audio/wav')
            
            with col2:
                st.markdown("**Enhanced Audio**")
                st.audio(audio_buffer, format='audio/wav')
            
            # Add some metrics if possible (would need reference audio)
            # st.subheader("Quality Metrics")
            # st.write("Note: Accurate metrics require a clean reference audio file")
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
    
    else:
        st.info("Please upload an audio file to get started")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Audio Denoiser App | Created with Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()