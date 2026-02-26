# Audio Compression with Deep Learning

Research-focused implementation of audio compression using deep learning techniques including autoencoders, variational autoencoders, and ResNet-based architectures.

## PRIVACY & ETHICS DISCLAIMER

**IMPORTANT: This is a research demonstration tool for educational purposes only.**

- This tool is designed for research and educational use
- No personal data is stored, transmitted, or processed beyond the session
- The compression models are not intended for production use
- **Misuse of audio compression for malicious purposes (e.g., voice cloning, deepfakes, biometric identification) is strictly prohibited**
- Users are responsible for complying with applicable laws and regulations
- The developers assume no responsibility for misuse of this technology
- This tool should not be used for any form of biometric identification or voice cloning

## Features

- **Multiple Model Architectures**: Conv1D Autoencoder, Variational Autoencoder (VAE), ResNet Autoencoder
- **Advanced Loss Functions**: Reconstruction, perceptual, spectral, and compression penalties
- **Comprehensive Evaluation**: SNR, PSNR, compression ratio, spectral distance metrics
- **Interactive Demo**: Streamlit-based web interface for real-time compression
- **Modern Tech Stack**: PyTorch 2.x, Python 3.10+, device-agnostic (CUDA/MPS/CPU)
- **Reproducible Research**: Deterministic seeding, comprehensive logging, checkpointing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Audio-Compression-with-Deep-Learning.git
cd Audio-Compression-with-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

1. **Train a model**:
```bash
python src/train/trainer.py --config configs/train_config.yaml
```

2. **Run evaluation**:
```bash
python src/eval/evaluator.py --config configs/eval_config.yaml
```

3. **Launch interactive demo**:
```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data directory
├── demo/                  # Interactive demo
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── assets/                # Generated assets
└── docs/                  # Documentation
```

## Model Architectures

### 1. Conv1D Autoencoder
- 1D convolutional encoder-decoder architecture
- Efficient for temporal audio patterns
- Configurable latent dimensions and channels

### 2. Variational Autoencoder (VAE)
- Probabilistic latent space modeling
- Learned compression with uncertainty quantification
- Beta parameter for KL divergence weighting

### 3. ResNet Autoencoder
- Residual connections for deeper networks
- Better gradient flow and training stability
- Configurable number of residual blocks

## Loss Functions

The compression models use combined loss functions:

- **Reconstruction Loss**: MSE between original and reconstructed audio
- **Perceptual Loss**: L1 loss on high-frequency components
- **Spectral Loss**: STFT-based frequency domain loss
- **Compression Penalty**: L1 regularization on latent representations
- **KL Divergence**: For VAE models (reconstruction vs. prior)

## Evaluation Metrics

- **SNR (Signal-to-Noise Ratio)**: Quality preservation
- **PSNR (Peak Signal-to-Noise Ratio)**: Peak quality measurement
- **Compression Ratio**: Size reduction achieved
- **Spectral Distance**: Frequency domain similarity
- **MSE/MAE**: Reconstruction error measures

## Configuration

Models and training can be configured via YAML files:

```yaml
model:
  type: conv1d  # Options: conv1d, vae, resnet
  latent_dim: 128
  channels: 64

data:
  sample_rate: 22050
  target_length: 22050
  normalize: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: adam
```

## Data Format

The system expects audio files in the following format:
- **Supported formats**: WAV, MP3, FLAC, M4A
- **Sample rate**: Configurable (default: 22050 Hz)
- **Length**: Configurable (default: 1 second)
- **Normalization**: Automatic normalization to [-1, 1] range

### Dataset Structure
```
data/
├── wav/                   # Audio files
├── meta.csv              # Metadata (auto-generated)
└── synthetic/             # Synthetic test data (auto-generated)
```

## Training

### Basic Training
```bash
python src/train/trainer.py
```

### Advanced Training Options
```bash
python src/train/trainer.py \
    --config configs/train_config.yaml \
    --resume checkpoints/last_model.pth
```

### Training Features
- **Automatic checkpointing**: Best and last model saves
- **TensorBoard logging**: Real-time training monitoring
- **Gradient clipping**: Prevents exploding gradients
- **Device detection**: Automatic CUDA/MPS/CPU selection
- **Synthetic data generation**: Automatic test data creation

## Evaluation

### Comprehensive Evaluation
```bash
python src/eval/evaluator.py --output_dir evaluation_results
```

### Evaluation Outputs
- **Leaderboard**: Model performance ranking
- **Metrics comparison**: Box plots and statistical analysis
- **Compression vs. quality**: Trade-off visualization
- **Detailed report**: Text summary of results

## Interactive Demo

The Streamlit demo provides:
- **Audio upload**: Support for multiple formats
- **Real-time compression**: Instant model inference
- **Visualization**: Waveform and latent space plots
- **Metrics display**: Real-time quality measurements
- **Audio playback**: Original and reconstructed audio

### Launch Demo
```bash
streamlit run demo/streamlit_app.py
```

## Development

### Code Quality
- **Type hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Pytest test suite

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## Performance Considerations

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for training
- **CPU**: Multi-core CPU for data loading
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: SSD recommended for data I/O

### Optimization Tips
- Use appropriate batch sizes for your hardware
- Enable mixed precision training for faster training
- Use multiple workers for data loading
- Monitor GPU memory usage during training

## Limitations and Known Issues

1. **Model Capacity**: Current models are designed for research, not production
2. **Audio Length**: Fixed-length processing (configurable)
3. **Real-time Performance**: Not optimized for real-time applications
4. **Compression Efficiency**: Focus on quality over extreme compression ratios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{audio_compression_dl,
  title={Audio Compression with Deep Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Audio-Compression-with-Deep-Learning}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Librosa team for audio processing utilities
- Streamlit team for the demo framework
- The open-source audio processing community

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the example configurations in `configs/`

---

**Remember**: This tool is for research and educational purposes only. Please use responsibly and in accordance with applicable laws and ethical guidelines.
# Audio-Compression-with-Deep-Learning
