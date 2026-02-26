#!/usr/bin/env python3
"""Script to run quick inference test."""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.utils import get_device, set_seed
from src.losses.compression_losses import AudioMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick inference test')
    parser.add_argument('--model', type=str, default='conv1d',
                       choices=['conv1d', 'vae', 'resnet'],
                       help='Model type to test')
    parser.add_argument('--input_length', type=int, default=22050,
                       help='Input audio length')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create model
    if args.model == 'conv1d':
        model = Conv1DAutoencoder(
            input_length=args.input_length,
            latent_dim=args.latent_dim,
            channels=64
        )
    elif args.model == 'vae':
        model = VariationalAutoencoder(
            input_length=args.input_length,
            latent_dim=args.latent_dim,
            channels=64
        )
    elif args.model == 'resnet':
        model = ResNetAutoencoder(
            input_length=args.input_length,
            latent_dim=args.latent_dim,
            channels=64,
            num_blocks=4
        )
    
    model.to(device)
    model.eval()
    
    logger.info(f"Created {args.model} model")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, args.input_length).to(device)
    logger.info(f"Input shape: {dummy_input.shape}")
    
    # Run inference
    with torch.no_grad():
        if args.model == 'vae':
            reconstructed, mu, logvar, latent = model(dummy_input)
            logger.info(f"VAE outputs - Reconstructed: {reconstructed.shape}, "
                       f"Mu: {mu.shape}, LogVar: {logvar.shape}, Latent: {latent.shape}")
        else:
            reconstructed, latent = model(dummy_input)
            logger.info(f"Outputs - Reconstructed: {reconstructed.shape}, Latent: {latent.shape}")
    
    # Compute basic metrics
    metrics = AudioMetrics(sample_rate=22050)
    
    original_size = args.input_length * 4  # 4 bytes per float32
    compressed_size = latent.numel() * 4
    
    test_metrics = metrics.compute_all_metrics(
        dummy_input.squeeze(),
        reconstructed.squeeze(),
        original_size,
        compressed_size
    )
    
    logger.info("Test Metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Quick inference test completed successfully!")


if __name__ == '__main__':
    main()
