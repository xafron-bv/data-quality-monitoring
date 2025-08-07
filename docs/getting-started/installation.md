# Installation Guide

This guide will walk you through installing the Data Quality Detection System.

## Prerequisites

- Operating System: Linux, macOS, or Windows (WSL)
- Python: 3.8+
- Memory: 8GB+ (16GB+ recommended for ML/LLM)
- Storage: 5GB+ free for models/data
- GPU (Optional): CUDA-capable GPU for ML/LLM acceleration

## Setup

1. Clone the repository and create a virtual environment (use your preferred tool)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional dev tools:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Note: Install a CUDA-enabled PyTorch from the official instructions for your platform.

## Model Downloads

- ML models (sentence transformers) are downloaded on first use
- LLM models (if used) are downloaded on demand

## Configuration

- Brand configs live in `brand_configs/`
- Optionally set environment variables in your shell profile:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Verification

List available commands and run help:

```bash
python main.py --help
python main.py single-demo --help
```

## Troubleshooting

- ImportError: ensure your virtual environment is active; reinstall requirements
- CUDA OOM: reduce batch sizes; run without LLM; CPU fallback
- Model downloads: check connectivity; pre-download into model cache
- Permissions: ensure write access to output directories

