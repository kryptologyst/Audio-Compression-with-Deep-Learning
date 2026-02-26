"""Tests for audio compression models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.losses.compression_losses import CompressionLoss, VAELoss, AudioMetrics
from src.data.audio_loader import AudioLoader, create_synthetic_dataset
from src.utils import get_device, set_seed


class TestCompressionModels:
    """Test compression models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.input_length = 22050
        self.batch_size = 2
        
        # Create dummy input
        self.dummy_input = torch.randn(self.batch_size, 1, self.input_length).to(self.device)
    
    def test_conv1d_autoencoder(self):
        """Test Conv1D autoencoder."""
        model = Conv1DAutoencoder(
            input_length=self.input_length,
            latent_dim=128,
            channels=64
        ).to(self.device)
        
        # Test forward pass
        reconstructed, latent = model(self.dummy_input)
        
        assert reconstructed.shape == self.dummy_input.shape
        assert latent.shape == (self.batch_size, 128)
        assert torch.all(torch.isfinite(reconstructed))
        assert torch.all(torch.isfinite(latent))
    
    def test_variational_autoencoder(self):
        """Test Variational Autoencoder."""
        model = VariationalAutoencoder(
            input_length=self.input_length,
            latent_dim=128,
            channels=64
        ).to(self.device)
        
        # Test forward pass
        reconstructed, mu, logvar, latent = model(self.dummy_input)
        
        assert reconstructed.shape == self.dummy_input.shape
        assert mu.shape == (self.batch_size, 128)
        assert logvar.shape == (self.batch_size, 128)
        assert latent.shape == (self.batch_size, 128)
        assert torch.all(torch.isfinite(reconstructed))
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))
        assert torch.all(torch.isfinite(latent))
    
    def test_resnet_autoencoder(self):
        """Test ResNet autoencoder."""
        model = ResNetAutoencoder(
            input_length=self.input_length,
            latent_dim=128,
            channels=64,
            num_blocks=4
        ).to(self.device)
        
        # Test forward pass
        reconstructed, latent = model(self.dummy_input)
        
        assert reconstructed.shape == self.dummy_input.shape
        assert latent.shape == (self.batch_size, 128)
        assert torch.all(torch.isfinite(reconstructed))
        assert torch.all(torch.isfinite(latent))


class TestLossFunctions:
    """Test loss functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.batch_size = 2
        self.input_length = 22050
        
        # Create dummy data
        self.original = torch.randn(self.batch_size, 1, self.input_length).to(self.device)
        self.reconstructed = torch.randn(self.batch_size, 1, self.input_length).to(self.device)
        self.latent = torch.randn(self.batch_size, 128).to(self.device)
    
    def test_compression_loss(self):
        """Test compression loss function."""
        criterion = CompressionLoss()
        
        loss, loss_dict = criterion(self.reconstructed, self.original, self.latent)
        
        assert torch.isfinite(loss)
        assert loss > 0
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
    
    def test_vae_loss(self):
        """Test VAE loss function."""
        criterion = VAELoss()
        
        mu = torch.randn(self.batch_size, 128).to(self.device)
        logvar = torch.randn(self.batch_size, 128).to(self.device)
        
        loss, loss_dict = criterion(self.reconstructed, self.original, mu, logvar)
        
        assert torch.isfinite(loss)
        assert loss > 0
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert 'kl_loss' in loss_dict


class TestAudioLoader:
    """Test audio loading utilities."""
    
    def test_audio_loader(self):
        """Test AudioLoader class."""
        loader = AudioLoader(sample_rate=22050, target_length=22050)
        
        # Create dummy audio
        audio = np.random.randn(44100)  # 2 seconds at 22050 Hz
        
        # Test preprocessing
        processed = loader.preprocess_audio(audio)
        
        assert len(processed) == 22050
        assert np.all(np.isfinite(processed))
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_create_synthetic_dataset(self, tmp_path):
        """Test synthetic dataset creation."""
        output_dir = tmp_path / "synthetic"
        
        create_synthetic_dataset(
            output_dir=output_dir,
            num_files=5,
            sample_rate=22050,
            duration=1.0
        )
        
        # Check that files were created
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 5
        
        # Check that files are valid
        for wav_file in wav_files:
            assert wav_file.exists()
            assert wav_file.stat().st_size > 0


class TestAudioMetrics:
    """Test audio metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = AudioMetrics(sample_rate=22050)
        
        # Create dummy audio
        self.original = torch.randn(1, 22050)
        self.reconstructed = self.original + 0.1 * torch.randn(1, 22050)
        self.original_size = 22050 * 4
        self.compressed_size = 128 * 4
    
    def test_compute_all_metrics(self):
        """Test metrics computation."""
        metrics = self.metrics.compute_all_metrics(
            self.original,
            self.reconstructed,
            self.original_size,
            self.compressed_size
        )
        
        assert isinstance(metrics, dict)
        assert 'snr_db' in metrics
        assert 'compression_ratio' in metrics
        assert 'spectral_distance' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'psnr_db' in metrics
        
        # Check that all metrics are finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"


if __name__ == "__main__":
    pytest.main([__file__])
