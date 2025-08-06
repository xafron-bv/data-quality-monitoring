# Installation Guide

This guide will walk you through the installation process for the Data Quality Detection System.

## Prerequisites

Before installing the system, ensure you have the following prerequisites:

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for ML/LLM models)
- **Storage**: At least 5GB free disk space for models and data
- **GPU** (Optional): CUDA-capable GPU for accelerated ML/LLM processing

### Software Dependencies

- Git (for cloning the repository)
- Python pip package manager
- Virtual environment tool (venv, conda, or virtualenv)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>  # The actual directory name will depend on your repository
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Using venv (built-in Python module)
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Alternative: Using conda
conda create -n data-quality python=3.8
conda activate data-quality
```

### 3. Install Core Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The core dependencies include:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning utilities
- `sentence-transformers`: ML-based detection models
- `torch`: Deep learning framework
- `transformers`: Hugging Face transformers library
- `datasets`: Dataset loading and processing
- `accelerate`: Hardware-accelerated training
- `matplotlib` & `seaborn`: Visualization tools
- `evaluate`: Model evaluation metrics

### 4. Install Development Dependencies (Optional)

If you plan to contribute or modify the code:

```bash
pip install -r requirements-dev.txt
```

This includes:
- `pre-commit`: Git hooks for code quality
- `flake8` and extensions: Comprehensive code linting
- `black`: Code formatting
- `isort`: Import sorting
- `mypy`: Type checking
- `pytest`: Testing framework
- `bandit`: Security scanning
- `sphinx`: Documentation generation

### 5. Install Pre-commit Hooks (Optional)

To ensure code quality on every commit:

```bash
pre-commit install
```

This sets up automatic code quality checks that run before each commit. To run the checks manually:

```bash
pre-commit run --all-files
```

## GPU Support Setup

For faster ML and LLM model processing:

### NVIDIA GPU with CUDA

1. Install CUDA Toolkit (11.7 or higher)
2. Install cuDNN (compatible with your CUDA version)
3. Install PyTorch with CUDA support:

```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Model Downloads

Some detection methods require pre-trained models:

### ML-Based Detection Models

The system will automatically download required models on first use:
- Sentence transformer models (~400MB each)
- Custom fine-tuned models (stored in `anomaly_detectors/ml_based/models/`)

### LLM-Based Detection Models

For LLM detection, models are downloaded on demand:
- Base transformer models (~1-2GB)
- Fine-tuned models (stored in `anomaly_detectors/llm_based/models/`)

## Configuration Setup

### 1. Brand Configuration

Create or modify brand configuration files:

```bash
# Create a new brand configuration based on the example
cp brand_configs/esqualo.json brand_configs/your_brand.json

# Edit with your brand's field mappings
vim brand_configs/your_brand.json
```

### 2. Environment Variables (Optional)

You can set environment variables to control system behavior:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# For persistent settings, add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
```

Note: The system doesn't currently use a .env file. Environment variables must be set in your shell or system environment.

## Verification

Verify your installation by running a simple detection demo:

```bash
# Run a simple detection demo
python main.py single-demo --help
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'X'**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again

2. **CUDA out of memory**
   - Reduce batch size in configuration
   - Use CPU mode: `--device cpu`

3. **Model download failures**
   - Check internet connection
   - Manually download models to the models directory

4. **Permission denied errors**
   - Ensure write permissions for output directories
   - Run with appropriate user permissions

### Getting Help

- Review error logs in the output directory
- Submit issues on the project repository

