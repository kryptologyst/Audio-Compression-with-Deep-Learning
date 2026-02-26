"""Audio data loading and preprocessing utilities."""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    """Audio loading and preprocessing class."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        target_length: Optional[int] = None,
        normalize: bool = True,
        pre_emphasis: float = 0.97
    ):
        """Initialize audio loader.
        
        Args:
            sample_rate: Target sample rate
            target_length: Target audio length in samples
            normalize: Whether to normalize audio
            pre_emphasis: Pre-emphasis coefficient
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.normalize = normalize
        self.pre_emphasis = pre_emphasis
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio signal.
        
        Args:
            audio: Input audio array
            
        Returns:
            Preprocessed audio array
        """
        # Pre-emphasis
        if self.pre_emphasis > 0:
            audio = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        
        # Normalize
        if self.normalize:
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Ensure target length
        if self.target_length is not None:
            audio = self._ensure_length(audio, self.target_length)
        
        return audio
    
    def _ensure_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Ensure audio has target length."""
        current_length = len(audio)
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            # Random crop for training, center crop for inference
            start = (current_length - target_length) // 2
            return audio[start:start + target_length]
        else:
            # Pad with zeros
            padding = np.zeros(target_length - current_length)
            return np.concatenate([audio, padding])
    
    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path], sample_rate: int) -> None:
        """Save audio to file.
        
        Args:
            audio: Audio array to save
            file_path: Output file path
            sample_rate: Sample rate
        """
        try:
            sf.write(file_path, audio, sample_rate)
            logger.info(f"Saved audio to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio {file_path}: {e}")
            raise


class AudioDataset:
    """Audio dataset class for compression training."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 22050,
        target_length: Optional[int] = None,
        normalize: bool = True,
        file_extensions: List[str] = None
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            target_length: Target audio length in samples
            normalize: Whether to normalize audio
            file_extensions: Allowed file extensions
        """
        self.data_dir = Path(data_dir)
        self.loader = AudioLoader(sample_rate, target_length, normalize)
        
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        self.file_extensions = file_extensions
        self.audio_files = self._find_audio_files()
        
        logger.info(f"Found {len(self.audio_files)} audio files in {data_dir}")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in data directory."""
        audio_files = []
        
        for ext in self.file_extensions:
            audio_files.extend(self.data_dir.rglob(f"*{ext}"))
            audio_files.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        
        return sorted(audio_files)
    
    def __len__(self) -> int:
        """Return number of audio files."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get audio file by index.
        
        Args:
            idx: File index
            
        Returns:
            Tuple of (audio_array, file_path)
        """
        file_path = self.audio_files[idx]
        audio, _ = self.loader.load_audio(file_path)
        audio = self.loader.preprocess_audio(audio)
        
        return audio, str(file_path)
    
    def create_metadata(self, output_path: Union[str, Path]) -> None:
        """Create metadata CSV file.
        
        Args:
            output_path: Path to save metadata CSV
        """
        metadata = []
        
        for file_path in self.audio_files:
            try:
                audio, sr = self.loader.load_audio(file_path)
                metadata.append({
                    'id': file_path.stem,
                    'path': str(file_path),
                    'sample_rate': sr,
                    'duration': len(audio) / sr,
                    'length': len(audio)
                })
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
        
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        logger.info(f"Metadata saved to: {output_path}")


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_files: int = 100,
    sample_rate: int = 22050,
    duration: float = 2.0,
    noise_level: float = 0.1
) -> None:
    """Create synthetic audio dataset for testing.
    
    Args:
        output_dir: Output directory
        num_files: Number of files to generate
        sample_rate: Sample rate
        duration: Duration in seconds
        noise_level: Noise level (0-1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = AudioLoader(sample_rate=sample_rate)
    
    for i in range(num_files):
        # Generate synthetic audio (mix of sine waves + noise)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Multiple sine waves with different frequencies
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
            0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
            0.2 * np.sin(2 * np.pi * 1320 * t)   # E6
        )
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save
        output_path = output_dir / f"synthetic_{i:03d}.wav"
        loader.save_audio(audio, output_path, sample_rate)
    
    logger.info(f"Created {num_files} synthetic audio files in {output_dir}")
