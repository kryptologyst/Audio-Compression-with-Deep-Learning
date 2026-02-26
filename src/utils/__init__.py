"""Core utilities for audio compression project."""

import random
import numpy as np
import torch
from typing import Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device
) -> Tuple[int, float]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        filepath: Path to checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def normalize_audio(audio: np.ndarray, target_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Normalize audio to target range.
    
    Args:
        audio: Input audio array
        target_range: Target range (min, max)
        
    Returns:
        Normalized audio array
    """
    min_val, max_val = target_range
    audio_min, audio_max = audio.min(), audio.max()
    
    if audio_max == audio_min:
        return np.zeros_like(audio)
    
    # Normalize to [0, 1] then scale to target range
    normalized = (audio - audio_min) / (audio_max - audio_min)
    return normalized * (max_val - min_val) + min_val


def ensure_length(audio: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """Ensure audio has target length by padding or truncating.
    
    Args:
        audio: Input audio array
        target_length: Target length
        pad_value: Value to use for padding
        
    Returns:
        Audio array with target length
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
    elif current_length > target_length:
        return audio[:target_length]
    else:
        padding = np.full(target_length - current_length, pad_value, dtype=audio.dtype)
        return np.concatenate([audio, padding])


def compute_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Compute compression ratio.
    
    Args:
        original_size: Original data size in bytes
        compressed_size: Compressed data size in bytes
        
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
