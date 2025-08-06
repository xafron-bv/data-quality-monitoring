# Installation and Setup Guide

## Prerequisites

Before installing the Data Quality Detection System, ensure you have the following:

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

# Using conda (if you prefer)
conda create -n data-quality python=3.8
conda activate data-quality
```

### 3. Install Core Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Development Dependencies (Optional)

For contributing to the project:

```bash
# Install development tools (linting, formatting, etc.)
pip install -r requirements-dev.txt

# Install pre-commit hooks for code quality checks
pre-commit install
```

This sets up automatic code quality checks that run before each commit, catching:
- Import errors and missing modules
- Syntax errors
- Basic code style issues

To run the checks manually: `pre-commit run --all-files`

### 5. Verify Installation

```bash
# Check installation
python main.py --help

# Run a simple test
python -c "import pandas as pd; import torch; print('Installation successful!')"
```

### 6. Download Sample Data (Optional)

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download sample datasets (if provided)
# wget <sample-data-url> -O data/sample_data.csv
```

## Quick Start Tutorial

### Your First Detection Run

The easiest way to start is with the single sample demo:

```bash
python main.py single-demo \
    --data-file data/sample_data.csv \
    --output-dir results/quick_start
```

This will:
1. Load your data
2. Inject synthetic anomalies for testing
3. Run all detection methods
4. Generate comprehensive reports

### Understanding the Output

#### Console Output
You'll see real-time progress:
```
🔍 Processing sample: quick_start_sample
📊 Total fields to check: 15
━━━━━━━━━━━━━━━━━━━━ 100% 15/15
✅ Detection complete!
```

#### Generated Files
Check your output directory:
- `report.json` - Detailed detection results
- `viewer_report.json` - Formatted for the web viewer
- `anomaly_summary.csv` - Summary of all detections
- `confusion_matrix/` - Performance visualizations

#### Web Viewer
1. Open `data_quality_viewer.html` in your browser
2. Upload the generated CSV and JSON files
3. Explore interactive visualizations

## Configuration Setup

### 1. Brand Configuration

Create or edit brand configuration files:

```bash
# Create brand configs directory if it doesn't exist
mkdir -p brand_configs

# Copy example configuration
cp brand_configs/example.json brand_configs/your_brand.json

# Edit configuration
nano brand_configs/your_brand.json
```

Example configuration:
```json
{
    "brand_name": "your_brand",
    "field_mappings": {
        "material": "Material_Column",
        "color_name": "Color_Description",
        "category": "Product_Category"
    },
    "default_data_path": "data/your_data.csv"
}
```

### 2. Environment Variables (Optional)

Create a `.env` file for environment-specific settings:

```bash
# Model cache directory
MODEL_CACHE_DIR=/path/to/model/cache

# GPU configuration
CUDA_VISIBLE_DEVICES=0

# Memory limits
MAX_WORKERS=4
BATCH_SIZE=32
```

### 3. Model Setup

#### Pre-trained Models

Download pre-trained models (if available):

```bash
# Create models directory
mkdir -p data/models

# Download models
# wget <model-url> -O data/models/model_name.pkl
```

#### Train Your Own Models

For ML-based detection:
```bash
python main.py ml-train \
    --data-file data/training_data.csv \
    --fields material color_name category
```

## Basic Usage Examples

### 1. Test on Clean Data
```bash
python main.py single-demo \
    --data-file clean_data.csv \
    --injection-intensity 0.0 \
    --output-dir results/baseline
```

### 2. Evaluate Detection Performance
```bash
python main.py single-demo \
    --data-file test_data.csv \
    --injection-intensity 0.2 \
    --generate-weights \
    --output-dir results/evaluation
```

### 3. Production Monitoring
```bash
python main.py single-demo \
    --data-file production_data.csv \
    --injection-intensity 0.0 \
    --use-weighted-combination \
    --weights-file config/production_weights.json \
    --output-dir results/monitoring
```

## Detection Methods Configuration

### Enable Specific Methods

Choose which detection methods to use:

```bash
# Validation only (fastest)
python main.py single-demo \
    --data-file your_data.csv \
    --enable-validation

# Pattern-based detection
python main.py single-demo \
    --data-file your_data.csv \
    --enable-pattern

# ML-based detection (requires trained models)
python main.py single-demo \
    --data-file your_data.csv \
    --enable-ml

# All methods
python main.py single-demo \
    --data-file your_data.csv \
    --enable-validation \
    --enable-pattern \
    --enable-ml \
    --enable-llm
```

### Adjust Detection Sensitivity

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.7 \
    --ml-threshold 0.8 \
    --llm-threshold 0.6
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory
**Symptoms**: Process killed, memory errors
**Solutions**:
- Use `--core-fields-only` flag
- Reduce batch size: `--batch-size 16`
- Disable ML/LLM detection
- Process fields sequentially

#### 2. Slow Performance
**Symptoms**: Long processing times
**Solutions**:
- Enable GPU: Check CUDA installation
- Use parallel processing: `--max-workers 8`
- Optimize thresholds to skip unnecessary checks
- Use validation-only for quick checks

#### 3. Missing Fields
**Symptoms**: "Field not found" errors
**Solutions**:
- Check field mappings in your brand configuration file (e.g., `brand_configs/esqualo.json`)
- Verify column names in your data
- Use `analyze-column` to inspect data

#### 4. Model Loading Errors
**Symptoms**: "Model not found" or loading failures
**Solutions**:
- Ensure models are downloaded/trained
- Check model paths in configuration
- Verify PyTorch/TensorFlow installation

### Installation Verification

Run these checks if you encounter issues:

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "pandas|numpy|torch|scikit-learn"

# Test imports
python -c "
import pandas as pd
import numpy as np
import torch
import sklearn
print('All imports successful!')
"

# Check GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Platform-Specific Notes

### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, use ARM64 compatible packages

### Windows
- Use WSL2 for best compatibility
- Or use Anaconda for easier package management
- Path separators: Use forward slashes or raw strings

### Linux
- May need to install python3-dev: `sudo apt-get install python3-dev`
- For GPU support, ensure CUDA toolkit is installed

## Next Steps

Now that you have the system installed and configured:

1. **Explore Commands**: See [Command Line Usage](03-command-line-usage.md)
2. **Understand Architecture**: Read [Architecture and Design](04-architecture-design.md)
3. **Configure Your Brand**: Follow [Adding New Brands](08-adding-brands.md)
4. **Add Custom Fields**: Learn [Adding New Fields](07-adding-fields.md)

## Getting Help

If you continue to experience issues:
1. Check the [FAQ section](09-operations.md#faq)
2. Review error logs in the output directory
3. Consult the project's issue tracker
4. Reach out to the community support channels