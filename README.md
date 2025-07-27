# Data Quality Detection System

A comprehensive system for detecting data quality issues, anomalies, and validation errors in product data.

## Core Scripts

### 1. `single_sample_multi_field_demo.py`
**Purpose:** Test multiple detectors across all fields on a single data sample. Useful for quick testing and demonstration.

```bash
python3 single_sample_multi_field_demo.py \
  --brand esqualo \
  --data-file data/esqualo_2022_fall_original.csv \
  --output-dir demo_results \
  --injection-intensity 0.2 \
  --enable-validation \
  --enable-pattern \
  --enable-ml
```

### 2. `multi_sample_evaluation.py`
**Purpose:** Run statistical evaluation on a single field across multiple samples. Generates performance metrics and evaluation reports.

```bash
<<<<<<< HEAD
python3 multi_sample_evaluation.py \
  data/esqualo_2022_fall_original.csv \
  --field material \
  --brand esqualo \
  --output-dir evaluation_results \
  --run all \
  --num-samples 10
=======
# 1. Run evaluation to generate performance data
python3 single_sample_multi_field_demo.py --brand your_brand --injection-intensity 0.15 --core-fields-only --enable-validation --enable-pattern --enable-ml --enable-llm

# 2. Generate detection weights from performance results
python3 generate_detection_weights.py --input-file demo_results/demo_analysis_unified_report.json --output-file detection_weights.json --verbose

# 3. Run detection with weighted combination
python3 single_sample_multi_field_demo.py --brand your_brand --core-fields-only --enable-validation --enable-pattern --enable-ml --enable-llm --use-weighted-combination --weights-file detection_weights.json
>>>>>>> 29a76c1 (Refactor demo and evaluation scripts for clarity and focus)
```

## Utility Scripts

### 3. `analyze_column.py`
**Purpose:** Analyze unique values and statistics for a specific field in your data.

```bash
python3 analyze_column.py \
  data/esqualo_2022_fall_original.csv \
  --field material \
  --brand esqualo
```

### 4. `detection_comparison.py`
**Purpose:** Compare ML and LLM detection methods side-by-side with visualizations.

```bash
python3 detection_comparison.py \
  data/esqualo_2022_fall_original.csv \
  --fields material color_name \
  --brand esqualo \
  --output-dir comparison_results
```

### 5. `ml_curve_generator.py`
**Purpose:** Generate detection performance curves for different thresholds and methods.

```bash
python3 ml_curve_generator.py \
  data/esqualo_2022_fall_original.csv \
  --brand esqualo \
  --detection-type ml \
  --fields material category \
  --thresholds 0.1 0.3 0.5 0.7 0.9
```

### 6. `generate_detection_weights.py`
**Purpose:** Generate optimized detection weights from evaluation results.

```bash
python3 generate_detection_weights.py \
  --input-file demo_results/demo_analysis_unified_report.json \
  --output-file detection_weights.json \
  --verbose
```

## Quick Start

1. Install dependencies:
```bash
pip3 install pandas numpy scikit-learn torch sentence-transformers matplotlib seaborn
```

2. Run a quick demo:
```bash
python3 single_sample_multi_field_demo.py --brand esqualo
```

3. Evaluate a specific field:
```bash
python3 multi_sample_evaluation.py data/your_data.csv --field material --brand your_brand
```

## Project Structure

```
├── multi_sample_evaluation.py      # Multi-sample statistical evaluation
├── single_sample_multi_field_demo.py # Single-sample multi-field testing
├── analyze_column.py               # Field analysis utility
├── detection_comparison.py         # Detection method comparison
├── ml_curve_generator.py          # Performance curve generation
├── generate_detection_weights.py   # Weight optimization utility
├── validators/                     # Field-specific validators
├── anomaly_detectors/             # ML and LLM-based detectors
├── brand_configs/                 # Brand-specific configurations
└── data/                          # Data files
```

## Detection Methods

- **Validation**: Rule-based validation for each field type
- **Pattern Detection**: Statistical anomaly detection
- **ML Detection**: Machine learning-based anomaly detection
- **LLM Detection**: Large language model-based detection
- **Weighted Combination**: Optimized combination of all methods

## Configuration

### Brand Configuration
Create a JSON file in `brand_configs/` for your brand:

```json
{
  "brand_name": "your_brand",
  "field_mappings": {
<<<<<<< HEAD
    "material": "Material",
    "color_name": "Color"
=======
    "material": "ProductComposition",
    "color_name": "ColorDescription",
    "size": "SizeName"
  },
  "default_data_path": "data/mybrand_data.csv",
  "training_data_path": "data/mybrand_training.csv",
  
  "enabled_fields": ["material", "color_name", "size"]
}
```

### Using Different Brands

Specify the brand when running evaluations or demos:

```bash
# Run evaluation for a specific brand
python multi_sample_evaluation.py data/mybrand_data.csv --field material --brand mybrand

# Run demo with brand configuration
python single_sample_multi_field_demo.py --brand mybrand

# Train ML models for a specific brand
cd anomaly_detectors/ml_based
python index.py ../../data/mybrand_training.csv --brand mybrand
```

## 🧠 ML Model Training

### Training Overview

The ML-based anomaly detection uses **SentenceTransformers** with **triplet loss** training to learn semantic representations that distinguish between normal and anomalous data patterns.

### Training Commands

#### Train All Available Fields
```bash
cd anomaly_detectors/ml_based
python index.py ../../data/your_training_data.csv --brand your_brand
```

#### Train Specific Fields Only
```bash
cd anomaly_detectors/ml_based
python index.py ../../data/your_training_data.csv --brand your_brand --rules material color_name category
```

#### Train with Hyperparameter Optimization
```bash
cd anomaly_detectors/ml_based
python index.py ../../data/your_training_data.csv --brand your_brand --use-hp-search --hp-trials 20
```

#### Train Single Field with HP Search
```bash
cd anomaly_detectors/ml_based
python index.py ../../data/your_training_data.csv --brand your_brand --rules material --use-hp-search --hp-trials 15
```

### Available Training Fields

The system supports training models for these fields:

| Field Name | Column Name | Description |
|------------|-------------|-------------|
| `material` | `material` | Product material composition |
| `color_name` | `colour_name` | Color names and descriptions |
| `category` | `article_structure_name_2` | Product categories |
| `size` | `size_name` | Product sizes |
| `ean` | `EAN` | European Article Numbers |
| `article_number` | `article_number` | Internal article numbers |
| `colour_code` | `colour_code` | Color codes |
| `customs_tariff_number` | `customs_tariff_number` | Customs classification |
| `description_short_1` | `description_short_1` | Short descriptions |
| `long_description_nl` | `long_description_NL` | Long Dutch descriptions |
| `product_name_en` | `product_name_EN` | English product names |

### Training Output

Training creates organized results in `anomaly_detectors/ml_based/results/`:

```
results/
├── results_material/           # Trained model for material field
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── results_colour_name/        # Trained model for color_name field
└── summary/                    # Training summaries and HP search results
    ├── hp_search_summary.json
    └── hp_search_results_*.json
```

### Checking Training Progress

Monitor training with these indicators:
- **Console output** shows training progress and metrics
- **Results directory** gets populated with trained models
- **Summary files** contain performance metrics and optimal parameters

## 🔍 Running Evaluations

### Evaluation Overview

The evaluation system tests all three detection approaches:
- Validates performance using error injection
- Measures recall, precision, and F1 scores
- Generates comprehensive reports

### Basic Evaluation Commands

#### Evaluate Single Field with All Methods
```bash
python multi_sample_evaluation.py data/your_data.csv \
  --brand your_brand \
  --field="material" \
  --validator="material" \
  --ml-detector \
  --run="all" \
  --output-dir="evaluation_results/material_evaluation"
```

#### Evaluate with Validation + ML Detection Only
```bash
python multi_sample_evaluation.py data/your_data.csv \
  --brand your_brand \
  --field="color_name" \
  --validator="color_name" \
  --ml-detector \
  --run="validation" \
  --output-dir="evaluation_results/color_validation_ml"
```

#### Evaluate ML Detection Only
```bash
python multi_sample_evaluation.py data/your_data.csv \
  --brand your_brand \
  --field="category" \
  --validator="category" \
  --ml-detector \
  --run="ml" \
  --output-dir="evaluation_results/category_ml_only"
```

### Batch Evaluations

#### Run Multiple Evaluations (Validation + Anomaly + ML)

```bash
# Run all configured evaluations
./run_evaluations.sh
```

#### Run Anomaly Detection Only

```bash
./run_anomaly_detection.sh
```

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--field` | Target field name (mapped to CSV column) | Required |
| `--validator` | Validator name (defaults to field) | Field name |
| `--anomaly-detector` | Anomaly detector name | Validator name |
| `--ml-detector` | Enable ML-based detection | False |
| `--run` | What to run: validation/anomaly/ml/both/all | both |
| `--num-samples` | Number of test samples | 32 |
| `--max-errors` | Max errors per sample | 3 |
| `--output-dir` | Output directory | evaluation_results |
| `--ignore-fp` | Ignore false positives | False |

### Understanding Evaluation Output

#### Console Output
```
--- Evaluation Setup ---
Target field: 'material'
Using validator: 'material'
Using anomaly detector: 'material'
Using ML-based anomaly detection

Evaluator initialized with: Validator + Reporter, Anomaly Detector + Reporter, ML Detector

=== VALIDATION, ANOMALY DETECTION, AND ML DETECTION SUMMARY ===

Sample: sample_0
  - Precision: 0.85
  - Recall: 0.92
  - F1 Score: 0.88
  - True Positives: 2
  - False Positives: 0
  - False Negatives: 0
  - Anomalies Detected: 1
  - ML Issues Detected: 3

OVERALL VALIDATION METRICS:
  - Overall Precision: 0.85
  - Overall Recall: 0.92
  - Overall F1 Score: 0.88

OVERALL ANOMALY DETECTION METRICS:
  - Total Anomalies Detected: 15

OVERALL ML DETECTION METRICS:
  - Total ML Issues Detected: 45
```

#### Generated Files
```
evaluation_results/
├── material_evaluation/
│   ├── summary_report.txt          # Human-readable summary
│   ├── full_evaluation_results.json # Complete results in JSON
│   ├── sample_0.csv               # Generated test samples
│   ├── sample_1.csv
│   └── ...
└── color_validation_ml/
    ├── summary_report.txt
    └── ...
```

## 📁 Directory Structure

```
data-quality-monitoring/
├── README.md                       # This file
├── multi_sample_evaluation.py      # Multi-sample statistical evaluation script
├── single_sample_multi_field_demo.py # Single-sample demo across multiple fields
├── evaluator.py                    # Evaluation orchestration
├── unified_detection_interface.py  # Unified detection framework
├── error_injection.py             # Error injection for testing
├── run_evaluations.sh             # Batch evaluation script
├── run_anomaly_detection.sh       # Anomaly detection script
├── data/                          # Input data files
│   └── your_data.csv
├── validators/                    # Rule-based validators
│   ├── material/
│   ├── color_name/
│   └── ...
├── anomaly_detectors/
│   ├── rule_based/               # Pattern-based detectors
│   └── ml_based/                 # ML-based detection
│       ├── index.py              # ML training interface
│       ├── model_training.py     # Training pipeline
│       ├── check_anomalies.py    # Inference engine
│       ├── results/              # Trained models
│       └── optimal_params.json   # Optimized parameters
├── error_injection_rules/        # Error rules for testing
└── evaluation_results/           # Evaluation outputs
```

## ⚙️ Configuration

### ML Model Configuration

Edit `anomaly_detectors/ml_based/optimal_params.json` to adjust model parameters:

```json
{
  "material": {
    "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "triplet_margin": 1.0,
    "distance_metric": "COSINE",
    "batch_size": 48,
    "epochs": 3,
    "learning_rate": 5e-6
>>>>>>> 29a76c1 (Refactor demo and evaluation scripts for clarity and focus)
  }
}
```

### Detection Weights
Use the generated weights file for optimized detection:

```bash
python3 single_sample_multi_field_demo.py \
  --brand your_brand \
  --use-weighted-combination \
  --weights-file detection_weights.json
```

## Key Features

- Multiple detection methods working in parallel
- Configurable error injection for testing
- Performance metrics and visualizations
- Brand-specific field mappings
- Extensible validator and detector framework
