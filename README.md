# Data Quality Detection System

A comprehensive, multi-method data quality monitoring system for detecting errors and anomalies in structured data, with a focus on fashion/retail product catalogs.

## Overview

This system combines multiple detection approaches to identify data quality issues:

- **Validation (Rule-Based)**: High-confidence error detection using business rules
- **Pattern-Based Anomaly Detection**: Medium-confidence detection using pattern matching
- **ML-Based Detection**: Semantic similarity analysis using sentence transformers
- **LLM-Based Detection**: Advanced semantic understanding with language models

The system is designed to be field-agnostic and brand-independent, making it adaptable to various data domains.

## Key Features

- 🎯 **Multi-Method Detection**: Combines rule-based, pattern-based, and ML approaches
- 📊 **Comprehensive Evaluation**: Built-in metrics and confusion matrix analysis
- 🔧 **Configurable**: Field mappings and detection thresholds are customizable
- 💾 **Memory Efficient**: Sequential processing and model caching
- 📈 **Performance Optimization**: Weighted combination based on historical performance
- 🌐 **Brand Agnostic**: Supports multiple brands through configuration
- 📱 **Visual Interface**: HTML5 viewer for interactive result exploration

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster ML/LLM detection)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd detection-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure brand settings:
```bash
# Edit brand_config.json with your field mappings
```

## Project Structure

The project is organized with a clear separation between entrypoints, dependencies, and data:

```
.
├── multi_sample_evaluation.py      # Main evaluation entrypoint
├── analyze_column.py               # Column analysis tool
├── generate_detection_weights.py   # Weight generation for detection methods
├── llm_model_training.py          # LLM model training entrypoint
├── ml_index.py                    # ML model indexing and training
├── common/                        # Shared modules and utilities
│   ├── brand_config.py           # Brand configuration loader
│   ├── common_interfaces.py      # Common interfaces and data structures
│   ├── comprehensive_detector.py  # Main detection orchestrator
│   ├── consolidated_reporter.py   # Unified reporting
│   ├── debug_config.py           # Debug configuration
│   ├── error_injection.py        # Error injection for testing
│   ├── evaluator.py              # Evaluation framework
│   ├── exceptions.py             # Custom exceptions
│   ├── field_column_map.py       # Field to column mapping
│   ├── field_mapper.py           # Field mapping utilities
│   ├── ml_curve_generator.py     # ML performance curve generation
│   └── unified_detection_interface.py  # Unified detection interface
├── anomaly_detectors/             # Anomaly detection modules
│   ├── ml_based/                 # ML-based detection
│   ├── pattern_based/            # Pattern-based detection
│   └── llm_based/                # LLM-based detection
├── validators/                    # Business rule validators
├── brand_configs/                 # Brand-specific configurations
└── data/                         # Data directory (git-ignored)
```

### Report Directories

Each entrypoint script generates reports in its own subdirectory:
- `multi_sample_evaluation/reports/` - Evaluation results and metrics
- `analyze_column/reports/` - Column analysis results
- `generate_detection_weights/reports/` - Detection weight configurations
- `llm_model_training/reports/` - LLM training results
- `ml_index/reports/` - ML model training results

Each entrypoint directory contains a `viewer.html` file for browsing reports in its `reports/` subdirectory.

## Quick Start

### Running the Demo

The easiest way to see the system in action:

```bash
# Basic demo with all detection methods
python single_sample_multi_field_demo.py \
    --data-file your_data.csv \
    --enable-validation \
    --enable-pattern \
    --enable-ml
```

### Basic Usage

1. **Analyze your data**:
```bash
python analyze_column.py --data-file your_data.csv --column product_name
```

2. **Run detection**:
```bash
python single_sample_multi_field_demo.py \
    --data-file your_data.csv \
    --injection-intensity 0.2 \
    --output-dir results
```

3. **View results**:
- Open `data_quality_viewer.html` in your browser
- Upload the generated CSV and JSON files from the results directory

## Configuration

### Brand Configuration

Edit `brand_config.json` to map your data columns to standard fields:

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

### Detection Thresholds

Adjust detection sensitivity:

```bash
python single_sample_multi_field_demo.py \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.7 \
    --ml-threshold 0.7 \
    --llm-threshold 0.6
```

## Adding New Fields

### 1. Rule-Based Validation

Create a new validator:
```bash
mkdir validators/new_field
# Create validators/new_field/validate.py
# Create validators/new_field/error_messages.json
```

### 2. Pattern-Based Detection

Add pattern rules:
```bash
# Create anomaly_detectors/pattern_based/rules/new_field.json
```

### 3. ML-Based Detection

Train a model:
```bash
# Add field config in model_training.py
python anomaly_detectors/ml_based/model_training.py --field new_field
```

## Advanced Usage

### Performance Optimization

Generate optimized weights based on evaluation results:
```bash
python generate_detection_weights.py \
    -i results/report.json \
    -o detection_weights.json
```

Use weighted combination:
```bash
python single_sample_multi_field_demo.py \
    --use-weighted-combination \
    --weights-file detection_weights.json
```

### Batch Evaluation

For systematic performance evaluation:
```bash
python multi_sample_evaluation.py \
    --data-file your_data.csv \
    --num-samples 10 \
    --sample-size 1000
```

### ML Model Analysis

Analyze ML model performance:
```bash
python ml_curve_generator.py \
    --field material \
    --output-dir ml_analysis
```

## Architecture

The system follows a modular, layered architecture:

- **Entry Points**: User-facing scripts for different use cases
- **Orchestration**: Coordinates detection methods and manages workflow
- **Detection Methods**: Independent implementations of each detection approach
- **Core Services**: Shared utilities for configuration, mapping, and reporting
- **Data Layer**: Handles data I/O and storage

## Performance Considerations

- **Memory Usage**: The system processes fields sequentially to minimize memory footprint
- **GPU Acceleration**: ML and LLM detection can utilize GPU if available
- **Caching**: Models are cached to avoid redundant loading
- **Batch Processing**: Configurable batch sizes for optimal performance

## Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use `--core-fields-only` flag
   - Reduce batch size
   - Disable ML/LLM detection

2. **Slow Performance**:
   - Enable GPU acceleration
   - Use parallel processing
   - Optimize detection thresholds

3. **Missing Fields**:
   - Check field mappings in brand_config.json
   - Verify column names in your data

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Submit pull requests with clear descriptions

## License

[Specify your license here]

## Support

For issues, questions, or contributions, please [specify contact method].