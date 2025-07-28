# Project Entrypoints

This document describes all executable entrypoints in the Data Quality Monitoring System project and their organization.

## Overview

Entrypoints are Python scripts with `if __name__ == "__main__":` blocks that can be executed directly. They are organized based on their purpose:
- General utility and evaluation tools remain in the root directory
- Detector-specific tools are placed in their respective detector directories

## Root Directory Entrypoints

### 1. `analyze_column.py`
**Purpose**: Utility for analyzing specific columns in CSV files  
**Usage**: `python analyze_column.py <csv_file> [field_name] --brand <brand_name>`  
**Description**: Provides statistical analysis and unique value listings for specified fields

### 2. `detection_comparison.py`
**Purpose**: Compares performance of ML and LLM detection methods  
**Usage**: `python detection_comparison.py --help`  
**Description**: Generates comparison reports and visualizations for different detection methods

### 3. `multi_sample_evaluation.py`
**Purpose**: Comprehensive evaluation across multiple samples  
**Usage**: `python multi_sample_evaluation.py --help`  
**Description**: Runs full evaluation pipeline with error injection and performance metrics

### 4. `single_sample_multi_field_demo.py`
**Purpose**: Demonstration of the complete system on a single sample  
**Usage**: `python single_sample_multi_field_demo.py --help`  
**Description**: Shows how all detection methods work together on a single data sample

### 5. `generate_detection_weights.py`
**Purpose**: Generates detection weights from performance results  
**Usage**: `python generate_detection_weights.py --input-file <report.json> --output-file detection_weights.json`  
**Description**: Creates weighted combination parameters based on detector performance

### 6. `error_injection.py`
**Purpose**: Utility module with test capabilities for error injection  
**Usage**: Primarily imported by other modules, but can be run standalone for testing  
**Description**: Provides error injection functionality for data quality testing

## ML-Based Detector Entrypoints

Located in `/workspace/anomaly_detectors/ml_based/`:

### 1. `ml_index_generator.py` (renamed from `index.py`)
**Purpose**: Generates ML models and indexes for anomaly detection  
**Usage**: `python ml_index_generator.py <csv_file> [--use-hp-search] [--check-anomalies FIELD]`  
**Description**: Trains sentence transformer models for each field with hyperparameter optimization

### 2. `generate_centroids_for_existing_models.py`
**Purpose**: Generates centroid files for existing ML models  
**Usage**: `python generate_centroids_for_existing_models.py <csv_file>`  
**Description**: Creates centroid representations for trained models to improve detection

## LLM-Based Detector Entrypoints

Located in `/workspace/anomaly_detectors/llm_based/`:

### 1. `train_llm_model.py` (renamed from `llm_model_training.py`)
**Purpose**: Trains language models for anomaly detection  
**Usage**: `python train_llm_model.py <data_file> --field <field_name> [--epochs N]`  
**Description**: Fine-tunes language models for field-specific anomaly detection

## Other Module Entrypoints

### 1. `/workspace/anomaly_detectors/anomaly_injection.py`
**Purpose**: Creates semantic anomalies for testing  
**Usage**: Primarily imported by other modules, but can be run standalone for testing  
**Description**: Generates semantically unusual but technically valid values for testing

## Non-Entrypoint Important Files

These files are imported by entrypoints but are not entrypoints themselves:

- `ml_curve_generator.py`: Imported by `detection_comparison.py` for generating curves
- `evaluator.py`: Core evaluation logic imported by multiple entrypoints
- `comprehensive_detector.py`: Unified detection interface
- `unified_detection_interface.py`: Common interface for all detectors

## Running Entrypoints

All entrypoints can be run with `--help` to see available options:

```bash
python <entrypoint>.py --help
```

Most entrypoints require data files and brand configuration. Example:

```bash
# Analyze a column
python analyze_column.py data/esqualo_2022_fall.csv color_name --brand esqualo

# Run ML model training
cd anomaly_detectors/ml_based
python ml_index_generator.py ../../data/esqualo_2022_fall.csv --use-hp-search

# Run single sample demo
python single_sample_multi_field_demo.py --data-file data/esqualo_2022_fall.csv
```

## Dependencies

Before running any entrypoint, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
pip install matplotlib seaborn evaluate  # Additional dependencies
```