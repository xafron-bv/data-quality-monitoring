# Data Quality Detection System

A comprehensive ML-powered system for detecting data quality issues, anomalies, and validation errors in product data. The system combines rule-based validation, statistical pattern detection, machine learning embeddings, and fine-tuned language models to provide multi-layered data quality assurance.

## Architecture Overview

The system employs a unified detection pipeline that orchestrates multiple detection methods:

- **Validation Layer**: Rule-based checks for format compliance and business logic
- **Pattern Detection**: Statistical anomaly detection using field-specific patterns
- **ML Detection**: Embedding-based anomaly detection using Sentence Transformers
- **LLM Detection**: Context-aware detection using fine-tuned language models
- **Weighted Combination**: Optimized ensemble of all detection methods

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for ML/LLM acceleration)
- 16GB RAM minimum (32GB recommended for LLM models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-quality-detection
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

3. Install ML dependencies:
```bash
pip install -r anomaly_detectors/ml_based/requirements.txt
```

4. (Optional) Install LLM dependencies for advanced detection:
```bash
pip install transformers accelerate
```

### Configuration

1. **Brand Configuration**: Create a JSON file in `brand_configs/` for your brand:
```json
{
  "brand_name": "your_brand",
  "field_mappings": {
    "material": "Material Column Name",
    "color_name": "Color Column Name",
    "category": "Category Column Name"
  },
  "default_data_path": "data/your_brand_data.csv",
  "enabled_fields": ["material", "color_name", "category"]
}
```

2. **Detection Weights**: Generate optimized weights from evaluation results:
```bash
python generate_detection_weights.py \
  --input-file evaluation_results/unified_report.json \
  --output-file detection_weights.json
```

## Usage

### Quick Start

Run a comprehensive test on a single sample:
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --data-file data/your_data.csv \
  --output-dir demo_results \
  --enable-validation \
  --enable-pattern \
  --enable-ml \
  --enable-llm
```

### Core Scripts

#### 1. Single Sample Multi-Field Demo
Tests all detection methods across all fields on a single data sample:
```bash
python single_sample_multi_field_demo.py \
  --brand esqualo \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir results \
  --injection-intensity 0.2 \
  --use-weighted-combination \
  --weights-file detection_weights.json
```

#### 2. Multi-Sample Evaluation
Performs statistical evaluation on a specific field across multiple samples:
```bash
python multi_sample_evaluation.py \
  data/your_data.csv \
  --field material \
  --brand your_brand \
  --output-dir evaluation_results \
  --run all \
  --num-samples 100 \
  --injection-rate 0.3
```

#### 3. ML Model Training
Train field-specific ML models:
```bash
cd anomaly_detectors/ml_based
python model_training.py \
  --data-file ../../data/training_data.csv \
  --field material \
  --output-dir results \
  --epochs 10 \
  --use-gpu
```

#### 4. Detection Performance Analysis
Compare detection methods and generate performance curves:
```bash
python ml_curve_generator.py \
  data/your_data.csv \
  --brand your_brand \
  --detection-type all \
  --fields material color_name \
  --thresholds 0.1 0.3 0.5 0.7 0.9
```

### Utility Scripts

- **Field Analysis**: `analyze_column.py` - Analyze unique values and patterns
- **Detection Comparison**: `detection_comparison.py` - Compare ML vs LLM methods
- **Confusion Matrix Analysis**: `confusion_matrix_analyzer.py` - Detailed performance metrics

## Detection Methods

### 1. Validation (Rule-Based)
- Field-specific format validation
- Business logic constraints
- Required field checks
- See [docs/VALIDATION_METHODS.md](docs/VALIDATION_METHODS.md)

### 2. Pattern-Based Detection
- Statistical anomaly detection
- Domain-specific pattern matching
- Configurable through JSON rules
- See [docs/PATTERN_DETECTION.md](docs/PATTERN_DETECTION.md)

### 3. ML-Based Detection
- Sentence Transformer embeddings
- Centroid-based anomaly detection
- GPU-accelerated inference
- See [docs/ML_DETECTION.md](docs/ML_DETECTION.md)

### 4. LLM-Based Detection
- Fine-tuned language models
- Context-aware anomaly detection
- Dynamic encoding with temporal awareness
- See [docs/LLM_DETECTION.md](docs/LLM_DETECTION.md)

## Project Structure

```
├── anomaly_detectors/          # Anomaly detection implementations
│   ├── llm_based/             # LLM-based detection
│   ├── ml_based/              # ML embedding detection
│   └── pattern_based/         # Pattern-based detection
├── validators/                 # Field-specific validators
│   ├── care_instructions/
│   ├── category/
│   ├── color_name/
│   ├── material/
│   └── ...
├── brand_configs/             # Brand-specific configurations
├── data/                      # Data files
├── docs/                      # Documentation
├── comprehensive_detector.py   # Main detection orchestrator
├── evaluator.py              # Evaluation framework
└── field_mapper.py           # Field mapping utilities
```

## Advanced Features

### Error Injection
Controlled error injection for testing:
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --injection-intensity 0.3 \
  --injection-types validation anomaly
```

### Weighted Detection
Combine multiple detection methods with optimized weights:
```bash
python single_sample_multi_field_demo.py \
  --use-weighted-combination \
  --weights-file detection_weights.json
```

### Batch Processing
Process large datasets efficiently:
```bash
python multi_sample_evaluation.py \
  large_dataset.csv \
  --batch-size 1000 \
  --max-workers 8
```

## Performance Tuning

- **GPU Acceleration**: Enable with `--use-gpu` flag
- **Batch Size**: Adjust `--batch-size` based on memory
- **Parallelization**: Set `--max-workers` for CPU cores
- **Model Caching**: Models are cached automatically

## Troubleshooting

1. **Memory Issues**: Reduce batch size or disable LLM detection
2. **GPU Errors**: Check CUDA installation or use `--no-gpu`
3. **Missing Models**: Run training scripts first
4. **Import Errors**: Ensure all dependencies are installed

## Contributing

1. Add new field types in `validators/`
2. Implement new detection methods following `AnomalyDetectorInterface`
3. Add brand configurations in `brand_configs/`
4. Update documentation in `docs/`

## License

[Your License Here]
