#!/usr/bin/env python3
"""Summary script to demonstrate the complete audio compression pipeline."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.losses.compression_losses import CompressionLoss, VAELoss, AudioMetrics
from src.data.audio_loader import AudioLoader, create_synthetic_dataset
from src.utils import get_device, set_seed, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the complete audio compression pipeline."""
    print("🎵 Audio Compression with Deep Learning - Complete Demo")
    print("=" * 60)
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\n1. Creating synthetic audio dataset...")
    data_dir = Path("data/wav")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    create_synthetic_dataset(
        output_dir=data_dir,
        num_files=5,
        sample_rate=22050,
        duration=1.0,
        noise_level=0.1
    )
    print(f"✓ Created 5 synthetic audio files in {data_dir}")
    
    # Load audio
    print("\n2. Loading and preprocessing audio...")
    loader = AudioLoader(sample_rate=22050, target_length=22050)
    audio_files = list(data_dir.glob("*.wav"))
    audio_file = audio_files[0]
    
    audio, sr = loader.load_audio(audio_file)
    processed_audio = loader.preprocess_audio(audio)
    print(f"✓ Loaded audio: {len(processed_audio)} samples, {len(processed_audio)/sr:.2f}s")
    
    # Create models
    print("\n3. Creating compression models...")
    
    models = {
        'Conv1D Autoencoder': Conv1DAutoencoder(
            input_length=22050,
            latent_dim=128,
            channels=64
        ),
        'Variational Autoencoder': VariationalAutoencoder(
            input_length=22050,
            latent_dim=128,
            channels=64
        ),
        'ResNet Autoencoder': ResNetAutoencoder(
            input_length=22050,
            latent_dim=128,
            channels=64,
            num_blocks=4
        )
    }
    
    for name, model in models.items():
        model.to(device)
        param_count = count_parameters(model)
        print(f"✓ {name}: {param_count:,} parameters")
    
    # Test compression
    print("\n4. Testing compression...")
    audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    metrics = AudioMetrics(sample_rate=22050)
    
    results = {}
    original_size = len(processed_audio) * 4
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == 'Variational Autoencoder':
                reconstructed, mu, logvar, latent = model(audio_tensor)
            else:
                reconstructed, latent = model(audio_tensor)
        
        compressed_size = latent.numel() * 4
        model_metrics = metrics.compute_all_metrics(
            audio_tensor.squeeze(),
            reconstructed.squeeze(),
            original_size,
            compressed_size
        )
        
        results[name] = {
            'metrics': model_metrics,
            'reconstructed': reconstructed.squeeze().cpu().numpy(),
            'latent': latent.squeeze().cpu().numpy()
        }
        
        print(f"✓ {name}: SNR={model_metrics['snr_db']:.2f}dB, "
              f"Compression={model_metrics['compression_ratio']:.2f}x")
    
    # Display results
    print("\n5. Compression Results Summary:")
    print("-" * 40)
    print(f"{'Model':<25} {'SNR (dB)':<10} {'Compression':<12} {'PSNR (dB)':<10}")
    print("-" * 40)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<25} {metrics['snr_db']:<10.2f} {metrics['compression_ratio']:<12.2f} {metrics['psnr_db']:<10.2f}")
    
    # Create visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original audio
    axes[0, 0].plot(processed_audio)
    axes[0, 0].set_title('Original Audio')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Reconstructed audios
    colors = ['red', 'green', 'blue']
    for i, (name, result) in enumerate(results.items()):
        axes[0, 1].plot(result['reconstructed'], color=colors[i], alpha=0.7, label=name)
    axes[0, 1].set_title('Reconstructed Audio')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    
    # Latent representations
    for i, (name, result) in enumerate(results.items()):
        axes[1, 0].plot(result['latent'], color=colors[i], alpha=0.7, label=name)
    axes[1, 0].set_title('Latent Representations')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    
    # Compression vs Quality
    snr_values = [result['metrics']['snr_db'] for result in results.values()]
    compression_values = [result['metrics']['compression_ratio'] for result in results.values()]
    model_names = list(results.keys())
    
    axes[1, 1].scatter(compression_values, snr_values, c=colors, s=100)
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name, (compression_values[i], snr_values[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('SNR (dB)')
    axes[1, 1].set_title('Compression vs Quality Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    plt.savefig(assets_dir / "compression_demo.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {assets_dir / 'compression_demo.png'}")
    
    # Final summary
    print("\n7. Demo Complete!")
    print("=" * 60)
    print("✓ Synthetic dataset created")
    print("✓ Three compression models tested")
    print("✓ Compression metrics computed")
    print("✓ Visualization generated")
    print("\nNext steps:")
    print("- Run training: python src/train/trainer.py")
    print("- Run evaluation: python src/eval/evaluator.py")
    print("- Launch demo: streamlit run demo/streamlit_app.py")
    print("\nRemember: This is a research tool for educational purposes only!")


if __name__ == "__main__":
    main()
