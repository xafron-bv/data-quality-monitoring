# Entry Points Documentation

This document describes the organization of entry point scripts and their associated report directories.

## Structure Overview

Entry point scripts (modules not imported by any other module) remain in the project root for easy access. Each entry point has:
- A dedicated `reports/` subdirectory for output files
- A `viewer.html` file within the reports directory for browsing results
- All reports directories are git-ignored to keep the repository clean

## Entry Points

### 1. `multi_sample_evaluation.py`
- **Purpose:** Systematic evaluation of detection methods with multiple samples
- **Reports Directory:** `multi_sample_evaluation/reports/`
- **Usage:**
  ```bash
  python multi_sample_evaluation.py data/source.csv --field material --num-samples 10
  ```
- **Key Features:**
  - Generates multiple samples with injected errors
  - Evaluates all detection methods
  - Produces comprehensive performance metrics
  - Supports brand-specific configurations

### 2. `analyze_column.py`
- **Purpose:** Analyze value distribution and patterns in CSV columns
- **Reports Directory:** `analyze_column/reports/`
- **Usage:**
  ```bash
  python analyze_column.py --data-file data.csv --field color_name --brand esqualo
  ```
- **Key Features:**
  - Shows unique values and frequencies
  - Detects whitespace variations
  - Exports analysis to text files

### 3. `generate_detection_weights.py`
- **Purpose:** Generate optimized detection weights from performance results
- **Reports Directory:** `generate_detection_weights/reports/`
- **Usage:**
  ```bash
  python generate_detection_weights.py -i evaluation_results/report.json
  ```
- **Key Features:**
  - Analyzes F1 scores across detection methods
  - Generates field-specific weights
  - Outputs JSON configuration for weighted detection

### 4. `llm_model_training.py`
- **Purpose:** Train LLM models for anomaly detection
- **Reports Directory:** `llm_model_training/reports/`
- **Usage:**
  ```bash
  python llm_model_training.py data.csv --field material --num-epochs 3
  ```
- **Key Features:**
  - Fine-tunes language models on field-specific data
  - Supports custom hyperparameters
  - Saves trained models and metrics

### 5. `ml_index.py`
- **Purpose:** Train ML models using sentence transformers
- **Reports Directory:** `ml_index/reports/`
- **Usage:**
  ```bash
  python ml_index.py data.csv --brand esqualo --field material
  ```
- **Key Features:**
  - Trains similarity-based anomaly detectors
  - Supports hyperparameter search
  - Generates model checkpoints and summaries

## Viewing Reports

Each entry point's reports directory contains a `viewer.html` file that provides:
- A modern, responsive interface for browsing reports
- Automatic detection of generated files (HTML, JSON, CSV)
- In-browser viewing with modal overlays
- Date extraction from filenames for organization

To view reports:
1. Navigate to the entry point's reports directory
2. Open `viewer.html` in a web browser
3. Click on any report card to view its contents

## Common Dependencies

All entry points utilize shared modules from the `common/` directory:
- `brand_config.py` - Brand configuration management
- `field_mapper.py` - Field to column mapping
- `evaluator.py` - Evaluation framework
- `comprehensive_detector.py` - Detection orchestration
- `error_injection.py` - Error injection utilities
- And more...

## Best Practices

1. **Output Organization:** Always use the designated reports directory for output
2. **Naming Convention:** Include dates in output filenames for easy tracking
3. **Configuration:** Use brand configurations for field mappings
4. **Memory Management:** The system handles large datasets through batching
5. **GPU Usage:** ML/LLM detection will automatically use GPU if available

## Extending the System

To add a new entry point:
1. Ensure it's not imported by any other module
2. Create a subdirectory with the same name
3. Add a `reports/` subdirectory
4. Copy and customize a `viewer.html` from another entry point
5. Update the output paths to use the reports directory
6. Add the reports directory to `.gitignore`