"""Evaluation script for audio compression models."""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.losses.compression_losses import AudioMetrics
from src.data.audio_loader import AudioDataset
from src.utils import get_device, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioCompressionEvaluator:
    """Evaluator class for audio compression models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = get_device()
        self.metrics = AudioMetrics(sample_rate=config['data']['sample_rate'])
        
        # Load models
        self.models = self._load_models()
        
        # Load test dataset
        self.test_dataset = self._load_test_dataset()
        
        logger.info(f"Loaded {len(self.models)} models for evaluation")
        logger.info(f"Test dataset contains {len(self.test_dataset)} audio files")
    
    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all available models."""
        models = {}
        
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
                        input_length=self.config['data']['target_length'],
                        latent_dim=config['latent_dim'],
                        channels=config['channels']
                    )
                elif config['type'] == 'vae':
                    model = VariationalAutoencoder(
                        input_length=self.config['data']['target_length'],
                        latent_dim=config['latent_dim'],
                        channels=config['channels'],
                        beta=config['beta']
                    )
                elif config['type'] == 'resnet':
                    model = ResNetAutoencoder(
                        input_length=self.config['data']['target_length'],
                        latent_dim=config['latent_dim'],
                        channels=config['channels'],
                        num_blocks=config['num_blocks']
                    )
                
                model.eval()
                model.to(self.device)
                models[name] = model
                logger.info(f"Loaded {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        
        return models
    
    def _load_test_dataset(self) -> AudioDataset:
        """Load test dataset."""
        return AudioDataset(
            data_dir=self.config['data']['test_dir'],
            sample_rate=self.config['data']['sample_rate'],
            target_length=self.config['data']['target_length'],
            normalize=self.config['data']['normalize']
        )
    
    def evaluate_model(self, model: torch.nn.Module, model_name: str) -> Dict[str, List[float]]:
        """Evaluate a single model.
        
        Args:
            model: Model to evaluate
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics for each test sample
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            'snr_db': [],
            'compression_ratio': [],
            'spectral_distance': [],
            'mse': [],
            'mae': [],
            'psnr_db': []
        }
        
        with torch.no_grad():
            for i in tqdm(range(len(self.test_dataset)), desc=f"Evaluating {model_name}"):
                audio, file_path = self.test_dataset[i]
                
                # Convert to tensor
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                audio_tensor = audio_tensor.to(self.device)
                
                # Forward pass
                if model_name == 'Variational Autoencoder':
                    reconstructed, mu, logvar, latent = model(audio_tensor)
                else:
                    reconstructed, latent = model(audio_tensor)
                
                # Compute metrics
                original_size = len(audio) * 4  # 4 bytes per float32
                compressed_size = latent.numel() * 4
                
                metrics = self.metrics.compute_all_metrics(
                    audio_tensor.squeeze(),
                    reconstructed.squeeze(),
                    original_size,
                    compressed_size
                )
                
                # Store results
                for key, value in metrics.items():
                    results[key].append(value)
        
        return results
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, List[float]]]:
        """Evaluate all models.
        
        Returns:
            Dictionary of results for each model
        """
        all_results = {}
        
        for model_name, model in self.models.items():
            results = self.evaluate_model(model, model_name)
            all_results[model_name] = results
        
        return all_results
    
    def create_leaderboard(self, results: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """Create leaderboard from evaluation results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            # Compute average metrics
            for metric_name, values in model_results.items():
                row[f'{metric_name}_mean'] = np.mean(values)
                row[f'{metric_name}_std'] = np.std(values)
            
            leaderboard_data.append(row)
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by SNR (higher is better)
        leaderboard = leaderboard.sort_values('snr_db_mean', ascending=False)
        
        return leaderboard
    
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, List[float]]], output_dir: Path):
        """Plot metrics comparison across models.
        
        Args:
            results: Evaluation results
            output_dir: Output directory for plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to plot
        metrics_to_plot = ['snr_db', 'compression_ratio', 'spectral_distance', 'psnr_db']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 8))
            
            model_names = []
            metric_values = []
            
            for model_name, model_results in results.items():
                if metric in model_results:
                    model_names.append(model_name)
                    metric_values.append(model_results[metric])
            
            # Create box plot
            plt.boxplot(metric_values, labels=model_names)
            plt.title(f'{metric.upper()} Comparison Across Models')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_compression_vs_quality(self, results: Dict[str, Dict[str, List[float]]], output_dir: Path):
        """Plot compression ratio vs quality trade-off.
        
        Args:
            results: Evaluation results
            output_dir: Output directory for plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, model_results) in enumerate(results.items()):
            if 'compression_ratio' in model_results and 'snr_db' in model_results:
                compression_ratios = model_results['compression_ratio']
                snr_values = model_results['snr_db']
                
                plt.scatter(compression_ratios, snr_values, 
                           label=model_name, color=colors[i % len(colors)], alpha=0.7)
        
        plt.xlabel('Compression Ratio')
        plt.ylabel('SNR (dB)')
        plt.title('Compression Ratio vs Quality Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'compression_vs_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, Dict[str, List[float]]], output_dir: Path):
        """Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create leaderboard
        leaderboard = self.create_leaderboard(results)
        
        # Save leaderboard
        leaderboard.to_csv(output_dir / 'leaderboard.csv', index=False)
        
        # Generate plots
        self.plot_metrics_comparison(results, output_dir)
        self.plot_compression_vs_quality(results, output_dir)
        
        # Generate summary report
        report_path = output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Audio Compression Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of models evaluated: {len(results)}\n")
            f.write(f"Number of test samples: {len(self.test_dataset)}\n")
            f.write(f"Sample rate: {self.config['data']['sample_rate']} Hz\n")
            f.write(f"Target length: {self.config['data']['target_length']} samples\n\n")
            
            f.write("Model Rankings (by SNR):\n")
            f.write("-" * 30 + "\n")
            
            for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
                f.write(f"{i}. {row['Model']}: {row['snr_db_mean']:.2f} ± {row['snr_db_std']:.2f} dB\n")
            
            f.write("\nDetailed Metrics:\n")
            f.write("-" * 20 + "\n")
            
            for _, row in leaderboard.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  SNR: {row['snr_db_mean']:.2f} ± {row['snr_db_std']:.2f} dB\n")
                f.write(f"  Compression Ratio: {row['compression_ratio_mean']:.2f} ± {row['compression_ratio_std']:.2f}x\n")
                f.write(f"  PSNR: {row['psnr_db_mean']:.2f} ± {row['psnr_db_std']:.2f} dB\n")
                f.write(f"  Spectral Distance: {row['spectral_distance_mean']:.4f} ± {row['spectral_distance_std']:.4f}\n")
        
        logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate audio compression models')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'data': {
                'test_dir': 'data/wav',
                'sample_rate': 22050,
                'target_length': 22050,
                'normalize': True
            }
        }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = AudioCompressionEvaluator(config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_all_models()
    
    # Generate report
    logger.info("Generating evaluation report...")
    evaluator.generate_report(results, output_dir)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
