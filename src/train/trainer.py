"""Training script for audio compression models."""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

from src.models.compression_models import Conv1DAutoencoder, VariationalAutoencoder, ResNetAutoencoder
from src.losses.compression_losses import CompressionLoss, VAELoss, AudioMetrics
from src.data.audio_loader import AudioDataset, create_synthetic_dataset
from src.utils import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioCompressionTrainer:
    """Trainer class for audio compression models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize metrics
        self.metrics = AudioMetrics(sample_rate=config['data']['sample_rate'])
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config['training']['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"Model has {count_parameters(self.model)} trainable parameters")
    
    def _create_model(self) -> nn.Module:
        """Create model based on config."""
        model_type = self.config['model']['type']
        
        if model_type == 'conv1d':
            return Conv1DAutoencoder(
                input_length=self.config['data']['target_length'],
                latent_dim=self.config['model']['latent_dim'],
                channels=self.config['model']['channels']
            )
        elif model_type == 'vae':
            return VariationalAutoencoder(
                input_length=self.config['data']['target_length'],
                latent_dim=self.config['model']['latent_dim'],
                channels=self.config['model']['channels'],
                beta=self.config['model'].get('beta', 1.0)
            )
        elif model_type == 'resnet':
            return ResNetAutoencoder(
                input_length=self.config['data']['target_length'],
                latent_dim=self.config['model']['latent_dim'],
                channels=self.config['model']['channels'],
                num_blocks=self.config['model'].get('num_blocks', 4)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on model type."""
        model_type = self.config['model']['type']
        
        if model_type == 'vae':
            return VAELoss(
                reconstruction_weight=self.config['loss']['reconstruction_weight'],
                kl_weight=self.config['loss']['kl_weight'],
                beta=self.config['model'].get('beta', 1.0)
            )
        else:
            return CompressionLoss(
                reconstruction_weight=self.config['loss']['reconstruction_weight'],
                perceptual_weight=self.config['loss']['perceptual_weight'],
                compression_weight=self.config['loss']['compression_weight'],
                use_spectral_loss=self.config['loss']['use_spectral_loss'],
                spectral_weight=self.config['loss']['spectral_weight']
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_data_loaders(self) -> tuple:
        """Create data loaders."""
        # Create synthetic dataset if no data directory exists
        data_dir = Path(self.config['data']['data_dir'])
        if not data_dir.exists() or len(list(data_dir.glob('*.wav'))) == 0:
            logger.info("Creating synthetic dataset...")
            create_synthetic_dataset(
                output_dir=data_dir,
                num_files=self.config['data'].get('num_synthetic_files', 100),
                sample_rate=self.config['data']['sample_rate'],
                duration=self.config['data'].get('duration', 2.0)
            )
        
        # Create dataset
        dataset = AudioDataset(
            data_dir=data_dir,
            sample_rate=self.config['data']['sample_rate'],
            target_length=self.config['data']['target_length'],
            normalize=self.config['data']['normalize']
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (audio, file_path) in enumerate(pbar):
            audio = audio.to(self.device)
            
            # Add channel dimension if needed
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config['model']['type'] == 'vae':
                reconstructed, mu, logvar, latent = self.model(audio)
                loss, loss_dict = self.criterion(reconstructed, audio, mu, logvar)
            else:
                reconstructed, latent = self.model(audio)
                loss, loss_dict = self.criterion(reconstructed, audio, latent)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['reconstruction_loss']:.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, global_step)
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss_dict[key] for loss_dict in epoch_losses])
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for audio, file_path in tqdm(self.val_loader, desc="Validation"):
                audio = audio.to(self.device)
                
                # Add channel dimension if needed
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                
                # Forward pass
                if self.config['model']['type'] == 'vae':
                    reconstructed, mu, logvar, latent = self.model(audio)
                    loss, loss_dict = self.criterion(reconstructed, audio, mu, logvar)
                else:
                    reconstructed, latent = self.model(audio)
                    loss, loss_dict = self.criterion(reconstructed, audio, latent)
                
                val_losses.append(loss_dict)
                
                # Compute metrics for a few samples
                if len(val_metrics) < 10:  # Limit to avoid memory issues
                    # Estimate compressed size (latent representation)
                    original_size = audio.numel() * 4  # 4 bytes per float32
                    compressed_size = latent.numel() * 4
                    
                    metrics = self.metrics.compute_all_metrics(
                        audio, reconstructed, original_size, compressed_size
                    )
                    val_metrics.append(metrics)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss_dict[key] for loss_dict in val_losses])
        
        # Average metrics
        if val_metrics:
            avg_metrics = {}
            for key in val_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in val_metrics])
            avg_losses.update(avg_metrics)
        
        return avg_losses
    
    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.4f}, "
                       f"Val Loss = {val_losses['total_loss']:.4f}")
            
            # Log to tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Save checkpoint
            if val_losses['total_loss'] < self.best_loss:
                self.best_loss = val_losses['total_loss']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_losses['total_loss'],
                    self.config['training']['checkpoint_dir'] / 'best_model.pth'
                )
            
            # Save last checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_losses['total_loss'],
                self.config['training']['checkpoint_dir'] / 'last_model.pth'
            )
        
        logger.info("Training completed!")
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        self.current_epoch, _ = load_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path,
            self.device
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train audio compression model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {
                'type': 'conv1d',
                'latent_dim': 128,
                'channels': 64,
                'beta': 1.0,
                'num_blocks': 4
            },
            'data': {
                'data_dir': 'data/wav',
                'sample_rate': 22050,
                'target_length': 22050,
                'normalize': True,
                'num_synthetic_files': 100,
                'duration': 2.0
            },
            'loss': {
                'reconstruction_weight': 1.0,
                'perceptual_weight': 0.1,
                'compression_weight': 0.01,
                'use_spectral_loss': True,
                'spectral_weight': 0.1,
                'kl_weight': 1.0
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'optimizer': 'adam',
                'grad_clip': 1.0,
                'log_interval': 100,
                'num_workers': 4,
                'log_dir': 'runs',
                'checkpoint_dir': 'checkpoints'
            },
            'seed': 42
        }
    
    # Create directories
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['data_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = AudioCompressionTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
