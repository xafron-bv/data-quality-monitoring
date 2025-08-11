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

## Data Management

### Encrypted Data Files

The project includes shell scripts to encrypt sensitive CSV data files for secure version control using native `zip` command with password protection. This allows you to commit encrypted data files to Git while keeping the original data secure.

#### Encrypting Data Files

To encrypt CSV files in the `data/` directory:

```bash
cd data
./encrypt.sh <password>
```

This will:
- Find all `.csv` files in the current directory
- Create a password-protected `encrypted_csv_files.zip` containing all CSV files
- Use native zip encryption for better compression and security

#### Decrypting Data Files

To decrypt and restore the original CSV files:

```bash
cd data
./decrypt.sh <password>
```

This will:
- Extract files from `encrypted_csv_files.zip` using the provided password
- Restore the original CSV files to the current directory

#### Workflow for New Clones

When setting up a new local clone of the repository:

1. **Clone the repository** (encrypted files are already committed)
2. **Navigate to data directory**:
   ```bash
   cd data
   ```
3. **Decrypt the data files**:
   ```bash
   ./decrypt.sh <password>
   ```
4. **Verify files are restored**:
   ```bash
   ls -la *.csv
   ```

#### Security Notes

- **Password Management**: Store the encryption password securely (e.g., in a password manager)
- **File Size**: Native zip compression provides better file size management
- **Backup**: Always keep a backup of the original data files
- **Git Ignore**: The original CSV files should be in `.gitignore` to prevent accidental commits

#### Troubleshooting

- **Wrong Password**: If you get decryption errors, verify the password is correct
- **Missing Files**: Ensure `encrypted_csv_files.zip` exists in the `data/` directory
- **Permissions**: Make sure the scripts are executable (`chmod +x encrypt.sh decrypt.sh`)
- **Zip Command**: Ensure `zip` and `unzip` commands are available on your system

## Verification

List available commands and run help:

```bash
python main.py --help
python main.py single-demo --help
```

## Troubleshooting

- ImportError: ensure your virtual environment is active; reinstall requirements
- CUDA OOM: reduce batch sizes; run without LLM; use `--device cpu` explicitly for LLM training
- Model downloads: check connectivity; pre-download into model cache
- Permissions: ensure write access to output directories

