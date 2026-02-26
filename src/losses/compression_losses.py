"""Loss functions and metrics for audio compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CompressionLoss(nn.Module):
    """Combined loss function for audio compression."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        compression_weight: float = 0.01,
        use_spectral_loss: bool = True,
        spectral_weight: float = 0.1
    ):
        """Initialize compression loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            compression_weight: Weight for compression penalty
            use_spectral_loss: Whether to use spectral loss
            spectral_weight: Weight for spectral loss
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.compression_weight = compression_weight
        self.use_spectral_loss = use_spectral_loss
        self.spectral_weight = spectral_weight
        
        # Perceptual loss using L1 on high-frequency components
        self.perceptual_loss = nn.L1Loss()
        
    def spectral_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss using STFT.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Spectral loss value
        """
        # Compute STFT
        x_stft = torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
        
        # Compute magnitude
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # Spectral loss
        spectral_loss = F.mse_loss(x_mag, y_mag)
        
        return spectral_loss
    
    def perceptual_loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss focusing on high frequencies.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Perceptual loss value
        """
        # High-pass filter to focus on high frequencies
        kernel = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        x_high = F.conv1d(x, kernel, padding=1)
        y_high = F.conv1d(y, kernel, padding=1)
        
        return self.perceptual_loss(x_high, y_high)
    
    def compression_penalty(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute compression penalty to encourage sparse representations.
        
        Args:
            latent: Latent representation
            
        Returns:
            Compression penalty value
        """
        # L1 penalty on latent representation
        l1_penalty = torch.mean(torch.abs(latent))
        
        # Entropy penalty to encourage diverse latent codes
        latent_prob = F.softmax(latent, dim=-1)
        entropy = -torch.sum(latent_prob * torch.log(latent_prob + 1e-8), dim=-1)
        entropy_penalty = -torch.mean(entropy)  # Negative entropy to encourage diversity
        
        return l1_penalty + 0.1 * entropy_penalty
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            reconstructed: Reconstructed audio
            original: Original audio
            latent: Latent representation
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss_fn(reconstructed, original)
        
        # Compression penalty
        compression_penalty = self.compression_penalty(latent)
        
        # Spectral loss
        spectral_loss = torch.tensor(0.0, device=reconstructed.device)
        if self.use_spectral_loss:
            spectral_loss = self.spectral_loss(reconstructed, original)
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.perceptual_weight * perceptual_loss +
            self.compression_weight * compression_penalty +
            self.spectral_weight * spectral_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'compression_penalty': compression_penalty.item(),
            'spectral_loss': spectral_loss.item()
        }
        
        return total_loss, loss_dict


class VAELoss(nn.Module):
    """Loss function for Variational Autoencoder."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta: float = 1.0
    ):
        """Initialize VAE loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence loss
            beta: Beta parameter for KL divergence
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.beta = beta
        
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss.
        
        Args:
            reconstructed: Reconstructed audio
            original: Original audio
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, original)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.beta * self.kl_weight * kl_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, loss_dict


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio.
    
    Args:
        original: Original audio
        reconstructed: Reconstructed audio
        
    Returns:
        SNR in dB
    """
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Compute compression ratio.
    
    Args:
        original_size: Original data size in bytes
        compressed_size: Compressed data size in bytes
        
    Returns:
        Compression ratio
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size


def compute_spectral_distance(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute spectral distance between original and reconstructed audio.
    
    Args:
        original: Original audio
        reconstructed: Reconstructed audio
        
    Returns:
        Spectral distance
    """
    # Compute STFT
    original_stft = torch.stft(original.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
    reconstructed_stft = torch.stft(reconstructed.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
    
    # Compute magnitude
    original_mag = torch.abs(original_stft)
    reconstructed_mag = torch.abs(reconstructed_stft)
    
    # Compute spectral distance
    spectral_distance = F.mse_loss(original_mag, reconstructed_mag)
    
    return spectral_distance.item()


class AudioMetrics:
    """Audio quality metrics calculator."""
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize metrics calculator.
        
        Args:
            sample_rate: Sample rate for audio
        """
        self.sample_rate = sample_rate
    
    def compute_all_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        original_size: int,
        compressed_size: int
    ) -> dict:
        """Compute all audio quality metrics.
        
        Args:
            original: Original audio
            reconstructed: Reconstructed audio
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # SNR
        metrics['snr_db'] = compute_snr(original, reconstructed)
        
        # Compression ratio
        metrics['compression_ratio'] = compute_compression_ratio(original_size, compressed_size)
        
        # Spectral distance
        metrics['spectral_distance'] = compute_spectral_distance(original, reconstructed)
        
        # MSE
        metrics['mse'] = F.mse_loss(original, reconstructed).item()
        
        # MAE
        metrics['mae'] = F.l1_loss(original, reconstructed).item()
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = metrics['mse']
        if mse > 0:
            max_val = torch.max(original).item()
            metrics['psnr_db'] = 20 * np.log10(max_val / np.sqrt(mse))
        else:
            metrics['psnr_db'] = float('inf')
        
        return metrics
