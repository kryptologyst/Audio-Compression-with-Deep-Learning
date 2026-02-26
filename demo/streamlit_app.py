"""Streamlit demo for audio compression."""

import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
from pathlib import Path
import logging

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.losses.compression_losses import AudioMetrics
from src.utils import get_device, load_checkpoint
from src.data.audio_loader import AudioLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Audio Compression with Deep Learning",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
**PRIVACY & ETHICS DISCLAIMER**

This is a research demonstration of audio compression using deep learning techniques. 

**Important Notes:**
- This tool is for educational and research purposes only
- No personal data is stored or transmitted
- The compression models are not intended for production use
- Misuse of audio compression for malicious purposes (e.g., voice cloning, deepfakes) is strictly prohibited
- Users are responsible for complying with applicable laws and regulations
- The developers assume no responsibility for misuse of this technology
"""


class AudioCompressionDemo:
    """Audio compression demo class."""
    
    def __init__(self):
        """Initialize demo."""
        self.device = get_device()
        self.models = {}
        self.loader = AudioLoader(sample_rate=22050, target_length=22050)
        self.metrics = AudioMetrics(sample_rate=22050)
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load available models."""
        model_configs = {
            'Conv1D Autoencoder': {
                'type': 'conv1d',
                'latent_dim': 128,
                'channels': 64
            },
            'Variational Autoencoder': {
                'type': 'vae',
                'latent_dim': 128,
                'channels': 64,
                'beta': 1.0
            },
            'ResNet Autoencoder': {
                'type': 'resnet',
                'latent_dim': 128,
                'channels': 64,
                'num_blocks': 4
            }
        }
        
        for name, config in model_configs.items():
            try:
                if config['type'] == 'conv1d':
                    model = Conv1DAutoencoder(
                        input_length=22050,
                        latent_dim=config['latent_dim'],
                        channels=config['channels']
                    )
                elif config['type'] == 'vae':
                    model = VariationalAutoencoder(
                        input_length=22050,
                        latent_dim=config['latent_dim'],
                        channels=config['channels'],
                        beta=config['beta']
                    )
                elif config['type'] == 'resnet':
                    model = ResNetAutoencoder(
                        input_length=22050,
                        latent_dim=config['latent_dim'],
                        channels=config['channels'],
                        num_blocks=config['num_blocks']
                    )
                
                model.eval()
                self.models[name] = model
                logger.info(f"Loaded {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
    
    def compress_audio(self, audio: np.ndarray, model_name: str) -> dict:
        """Compress audio using specified model.
        
        Args:
            audio: Input audio array
            model_name: Name of model to use
            
        Returns:
            Dictionary with compression results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        # Preprocess audio
        processed_audio = self.loader.preprocess_audio(audio)
        
        # Convert to tensor
        audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            if model_name == 'Variational Autoencoder':
                reconstructed, mu, logvar, latent = model(audio_tensor)
            else:
                reconstructed, latent = model(audio_tensor)
        
        # Convert back to numpy
        reconstructed_audio = reconstructed.squeeze().cpu().numpy()
        
        # Compute metrics
        original_size = len(processed_audio) * 4  # 4 bytes per float32
        compressed_size = latent.numel() * 4
        
        metrics = self.metrics.compute_all_metrics(
            audio_tensor.squeeze(),
            reconstructed.squeeze(),
            original_size,
            compressed_size
        )
        
        return {
            'original_audio': processed_audio,
            'reconstructed_audio': reconstructed_audio,
            'latent': latent.cpu().numpy(),
            'metrics': metrics
        }


def main():
    """Main demo function."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Audio Compression with Deep Learning</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(DISCLAIMER)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize demo
    demo = AudioCompressionDemo()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    available_models = list(demo.models.keys())
    if not available_models:
        st.error("No models available. Please check model files.")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Compression Model",
        available_models,
        help="Choose the deep learning model for audio compression"
    )
    
    # Audio input options
    st.sidebar.subheader("Audio Input")
    input_method = st.sidebar.radio(
        "Input Method",
        ["Upload File", "Record Audio", "Generate Synthetic"]
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Audio")
        
        audio_data = None
        sample_rate = 22050
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload an audio file to compress"
            )
            
            if uploaded_file is not None:
                try:
                    # Load audio
                    audio_bytes = uploaded_file.read()
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file.flush()
                        
                        audio_data, sample_rate = librosa.load(tmp_file.name, sr=22050)
                        Path(tmp_file.name).unlink()  # Clean up
                    
                    st.success(f"Loaded audio: {len(audio_data)} samples, {len(audio_data)/sample_rate:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
        
        elif input_method == "Record Audio":
            st.info("Audio recording feature would be implemented here using streamlit-audio-recorder")
            # This would require additional package: pip install streamlit-audio-recorder
        
        elif input_method == "Generate Synthetic":
            if st.button("Generate Synthetic Audio"):
                # Generate synthetic audio
                duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0)
                frequency = st.slider("Frequency (Hz)", 100, 2000, 440)
                noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = (
                    0.7 * np.sin(2 * np.pi * frequency * t) +
                    noise_level * np.random.normal(0, 1, len(t))
                )
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                st.success(f"Generated synthetic audio: {len(audio_data)} samples")
        
        # Display input audio
        if audio_data is not None:
            # Waveform plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=audio_data,
                mode='lines',
                name='Input Audio',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="Input Audio Waveform",
                xaxis_title="Sample",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Audio player
            st.audio(audio_data, sample_rate=sample_rate)
    
    with col2:
        st.subheader("Compressed Audio")
        
        if audio_data is not None and st.button("Compress Audio", type="primary"):
            try:
                with st.spinner("Compressing audio..."):
                    results = demo.compress_audio(audio_data, selected_model)
                
                # Display metrics
                st.subheader("Compression Metrics")
                
                metrics = results['metrics']
                
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric("SNR (dB)", f"{metrics['snr_db']:.2f}")
                    st.metric("Compression Ratio", f"{metrics['compression_ratio']:.2f}x")
                
                with col_metric2:
                    st.metric("PSNR (dB)", f"{metrics['psnr_db']:.2f}")
                    st.metric("Spectral Distance", f"{metrics['spectral_distance']:.4f}")
                
                # Display reconstructed audio
                reconstructed_audio = results['reconstructed_audio']
                
                # Waveform comparison
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Original Audio", "Reconstructed Audio"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(y=audio_data, mode='lines', name='Original', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=reconstructed_audio, mode='lines', name='Reconstructed', line=dict(color='red')),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(title_text="Sample")
                fig.update_yaxes(title_text="Amplitude")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Audio player for reconstructed audio
                st.audio(reconstructed_audio, sample_rate=sample_rate)
                
                # Latent space visualization
                st.subheader("Latent Representation")
                latent = results['latent'].flatten()
                
                fig_latent = go.Figure()
                fig_latent.add_trace(go.Scatter(
                    y=latent,
                    mode='lines+markers',
                    name='Latent Vector',
                    line=dict(color='green')
                ))
                fig_latent.update_layout(
                    title="Latent Representation",
                    xaxis_title="Dimension",
                    yaxis_title="Value",
                    height=300
                )
                st.plotly_chart(fig_latent, use_container_width=True)
                
                # Download reconstructed audio
                st.subheader("Download")
                
                # Create audio file for download
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, reconstructed_audio, sample_rate, format='WAV')
                audio_bytes.seek(0)
                
                st.download_button(
                    label="Download Reconstructed Audio",
                    data=audio_bytes.getvalue(),
                    file_name=f"compressed_{selected_model.lower().replace(' ', '_')}.wav",
                    mime="audio/wav"
                )
                
            except Exception as e:
                st.error(f"Error during compression: {e}")
                logger.error(f"Compression error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Audio Compression with Deep Learning - Research Demo</p>
        <p>Built with PyTorch, Streamlit, and Librosa</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
