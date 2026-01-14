import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import io
import matplotlib.pyplot as plt
import noisereduce as nr
from pydub import AudioSegment
import time
from main import process_audio # type: ignore

# Set page config
st.set_page_config(
    page_title="Audio Denoiser",
    page_icon="🎧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stFileUploader>div>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stProgress>div>div>div>div {
        background-color: #4CAF50;
    }
    .stAudio {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("🎧 Audio Denoising Application")
st.markdown("""
Upload an audio file and apply various denoising techniques to improve its quality.
Supported formats: WAV, MP3, FLAC, OGG
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    denoise_method = st.selectbox(
        "Denoising Method",
        ["NoiseReduce", "Spectral Gating", "Wiener Filter"],
        help="Choose the denoising algorithm to apply"
    )
    
    if denoise_method == "Spectral Gating":
        st.subheader("Spectral Gating Parameters")
        n_fft = st.slider("FFT Size", 256, 4096, 2048, step=256)
        win_length = st.slider("Window Length", 128, 2048, 1024, step=128)
        hop_length = st.slider("Hop Length", 32, 512, 256, step=32)
        threshold = st.slider("Threshold (dB)", -60, 0, -20, step=1)
    
    elif denoise_method == "Wiener Filter":
        st.subheader("Wiener Filter Parameters")
        wiener_window = st.slider("Window Size", 3, 21, 5, step=2)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses various audio processing techniques to remove background noise from recordings.
    - **NoiseReduce**: Uses a noise profile to subtract noise
    - **Spectral Gating**: Attenuates frequencies below a threshold
    - **Wiener Filter**: Statistical approach for noise reduction
    """)

# File upload section
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg"],
    help="Upload an audio file for denoising"
    
)

def plot_waveform(y, sr, title):
    """Plot waveform of audio signal"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_spectrogram(y, sr, title):
    """Plot spectrogram of audio signal"""
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

def apply_noisereduce(y, sr):
    """Apply noisereduce algorithm"""
    # Assume first 0.5 seconds is noise (for demo purposes)
    noise_sample = y[:int(0.5 * sr)]
    return nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, stationary=False)

def apply_spectral_gating(y, sr, n_fft, win_length, hop_length, threshold):
    """Apply spectral gating noise reduction"""
    # Compute STFT
    stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    
    # Compute magnitude and phase
    magnitude, phase = librosa.magphase(stft)
    
    # Compute power in dB
    power = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Create mask
    mask = power > threshold
    mask = mask.astype(float)
    
    # Apply mask
    denoised_stft = stft * mask
    
    # Inverse STFT
    return librosa.istft(denoised_stft, win_length=win_length, hop_length=hop_length)

def apply_wiener_filter(y, wiener_window):
    """Apply Wiener filter noise reduction"""
    return signal.wiener(y, mysize=wiener_window)

def process_audio(uploaded_file, method, **kwargs):
    """Process the uploaded audio file with selected method"""
    # Read audio file
    audio_bytes = uploaded_file.read()
    
    # Convert to AudioSegment for format conversion
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), 
                                         format=uploaded_file.name.split('.')[-1])
    
    # Convert to WAV format for processing
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    # Load with librosa
    y, sr = librosa.load(wav_io, sr=None)
    
    # Apply selected denoising method
    if method == "NoiseReduce":
        processed_audio = apply_noisereduce(y, sr)
    elif method == "Spectral Gating":
        processed_audio = apply_spectral_gating(y, sr, **kwargs)
    elif method == "Wiener Filter":
        processed_audio = apply_wiener_filter(y, **kwargs)
    else:
        processed_audio = y
    
    return y, sr, processed_audio

if uploaded_file is not None:
    # Display original audio
    st.header("Original Audio")
    st.audio(uploaded_file, format='audio/wav')
    
    # Process the audio with a progress bar
    with st.spinner("Processing audio..."):
        progress_bar = st.progress(0)
        
        # Get parameters based on selected method
        kwargs = {}
        if denoise_method == "Spectral Gating":
            kwargs = {
                'n_fft': n_fft,
                'win_length': win_length,
                'hop_length': hop_length,
                'threshold': threshold
            }
        elif denoise_method == "Wiener Filter":
            kwargs = {'wiener_window': wiener_window}
        
        # Simulate progress
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Process audio
        y, sr, processed_audio = process_audio(uploaded_file, denoise_method, **kwargs)
    
    st.success("Audio processing complete!")
    
    # Display processed audio
    st.header("Processed Audio")
    
    # Convert processed audio to playable format
    processed_io = io.BytesIO()
    sf.write(processed_io, processed_audio, sr, format='WAV')
    processed_io.seek(0)
    
    st.audio(processed_io, format='audio/wav')
    
    # Visualization section
    st.header("Audio Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Audio")
        plot_waveform(y, sr, "Original Waveform")
        plot_spectrogram(y, sr, "Original Spectrogram")
    
    with col2:
        st.subheader("Processed Audio")
        plot_waveform(processed_audio, sr, "Processed Waveform")
        plot_spectrogram(processed_audio, sr, "Processed Spectrogram")
    
    # Download button for processed audio
    st.download_button(
        label="Download Processed Audio",
        data=processed_io,
        file_name=f"denoised_{uploaded_file.name}",
        mime="audio/wav"
    )
else:
    st.info("Please upload an audio file to get started.")