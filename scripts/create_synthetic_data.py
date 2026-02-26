#!/usr/bin/env python3
"""Script to create synthetic dataset for testing."""

import argparse
from pathlib import Path
import logging

from src.data.audio_loader import create_synthetic_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create synthetic audio dataset')
    parser.add_argument('--output_dir', type=str, default='data/wav',
                       help='Output directory for synthetic files')
    parser.add_argument('--num_files', type=int, default=100,
                       help='Number of files to generate')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Sample rate')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration in seconds')
    parser.add_argument('--noise_level', type=float, default=0.1,
                       help='Noise level (0-1)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {args.num_files} synthetic audio files...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sample rate: {args.sample_rate} Hz")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Noise level: {args.noise_level}")
    
    # Create synthetic dataset
    create_synthetic_dataset(
        output_dir=output_dir,
        num_files=args.num_files,
        sample_rate=args.sample_rate,
        duration=args.duration,
        noise_level=args.noise_level
    )
    
    logger.info("Synthetic dataset creation completed!")


if __name__ == '__main__':
    main()
