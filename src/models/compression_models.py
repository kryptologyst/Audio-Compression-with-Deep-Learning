"""Advanced audio compression models using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class Conv1DAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for audio compression."""
    
    def __init__(
        self,
        input_length: int = 22050,
        latent_dim: int = 128,
        channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1
    ):
        """Initialize Conv1D autoencoder.
        
        Args:
            input_length: Input audio length
            latent_dim: Latent representation dimension
            channels: Number of channels in conv layers
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(channels * 2, channels * 4, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(channels * 4, channels * 8, kernel_size, stride, padding),
            nn.ReLU(),
        )
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            encoder_output = self.encoder(dummy_input)
            encoder_size = encoder_output.numel()
        
        # Latent projection
        self.latent_proj = nn.Linear(encoder_size, latent_dim)
        
        # Decoder projection
        self.decoder_proj = nn.Linear(latent_dim, encoder_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels * 8, channels * 4, kernel_size, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose1d(channels * 4, channels * 2, kernel_size, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose1d(channels * 2, channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, 1, kernel_size, stride, padding),
            nn.Tanh()
        )
        
        self.encoder_output_shape = encoder_output.shape[1:]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input audio to latent representation.
        
        Args:
            x: Input audio tensor (batch_size, 1, length)
            
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Flatten and project to latent space
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.latent_proj(encoded_flat)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.
        
        Args:
            latent: Latent representation (batch_size, latent_dim)
            
        Returns:
            Reconstructed audio (batch_size, 1, length)
        """
        # Project back to encoder output size
        decoded_flat = self.decoder_proj(latent)
        
        # Reshape to encoder output shape
        decoded = decoded_flat.view(decoded_flat.size(0), *self.encoder_output_shape)
        
        # Decode
        reconstructed = self.decoder(decoded)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input audio tensor
            
        Returns:
            Tuple of (reconstructed_audio, latent_representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for audio compression with learned priors."""
    
    def __init__(
        self,
        input_length: int = 22050,
        latent_dim: int = 128,
        channels: int = 64,
        beta: float = 1.0
    ):
        """Initialize VAE.
        
        Args:
            input_length: Input audio length
            latent_dim: Latent representation dimension
            channels: Number of channels in conv layers
            beta: Beta parameter for KL divergence weighting
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels, channels * 2, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels * 2, channels * 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels * 4, channels * 8, 3, 2, 1),
            nn.ReLU(),
        )
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            encoder_output = self.encoder(dummy_input)
            encoder_size = encoder_output.numel()
        
        # Mean and log variance projections
        self.mu_proj = nn.Linear(encoder_size, latent_dim)
        self.logvar_proj = nn.Linear(encoder_size, latent_dim)
        
        # Decoder projection
        self.decoder_proj = nn.Linear(latent_dim, encoder_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels * 8, channels * 4, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels * 4, channels * 2, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels * 2, channels, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, 1, 3, 2, 1),
            nn.Tanh()
        )
        
        self.encoder_output_shape = encoder_output.shape[1:]
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input audio to latent distribution parameters.
        
        Args:
            x: Input audio tensor (batch_size, 1, length)
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encode
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # Get distribution parameters
        mu = self.mu_proj(encoded_flat)
        logvar = self.logvar_proj(encoded_flat)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent representation
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.
        
        Args:
            latent: Latent representation (batch_size, latent_dim)
            
        Returns:
            Reconstructed audio (batch_size, 1, length)
        """
        # Project back to encoder output size
        decoded_flat = self.decoder_proj(latent)
        
        # Reshape to encoder output shape
        decoded = decoded_flat.view(decoded_flat.size(0), *self.encoder_output_shape)
        
        # Decode
        reconstructed = self.decoder(decoded)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input audio tensor
            
        Returns:
            Tuple of (reconstructed_audio, mu, logvar, latent)
        """
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, mu, logvar, latent


class ResidualBlock1D(nn.Module):
    """1D Residual block for audio processing."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        """Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNetAutoencoder(nn.Module):
    """ResNet-based autoencoder for audio compression."""
    
    def __init__(
        self,
        input_length: int = 22050,
        latent_dim: int = 128,
        channels: int = 64,
        num_blocks: int = 4
    ):
        """Initialize ResNet autoencoder.
        
        Args:
            input_length: Input audio length
            latent_dim: Latent representation dimension
            channels: Number of channels
            num_blocks: Number of residual blocks
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Conv1d(1, channels, 7, 2, 3)
        self.encoder_bn = nn.BatchNorm1d(channels)
        self.encoder_relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock1D(channels) for _ in range(num_blocks)
        ])
        
        # Downsampling layers
        self.downsample = nn.Conv1d(channels, channels * 2, 3, 2, 1)
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            encoded = self._encode_forward(dummy_input)
            encoder_size = encoded.numel()
        
        # Latent projection
        self.latent_proj = nn.Linear(encoder_size, latent_dim)
        self.decoder_proj = nn.Linear(latent_dim, encoder_size)
        
        # Decoder
        self.decoder_conv = nn.ConvTranspose1d(channels * 2, channels, 3, 2, 1)
        self.decoder_bn = nn.BatchNorm1d(channels)
        self.decoder_relu = nn.ReLU(inplace=True)
        
        # Decoder residual blocks
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock1D(channels) for _ in range(num_blocks)
        ])
        
        # Final output layer
        self.final_conv = nn.ConvTranspose1d(channels, 1, 7, 2, 3)
        self.final_tanh = nn.Tanh()
        
        self.encoder_output_shape = encoded.shape[1:]
    
    def _encode_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        x = self.encoder_relu(self.encoder_bn(self.encoder_conv(x)))
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.downsample(x)
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input audio to latent representation."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        encoded = self._encode_forward(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.latent_proj(encoded_flat)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio."""
        decoded_flat = self.decoder_proj(latent)
        decoded = decoded_flat.view(decoded_flat.size(0), *self.encoder_output_shape)
        
        decoded = self.decoder_relu(self.decoder_bn(self.decoder_conv(decoded)))
        
        for block in self.decoder_blocks:
            decoded = block(decoded)
        
        reconstructed = self.final_tanh(self.final_conv(decoded))
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
