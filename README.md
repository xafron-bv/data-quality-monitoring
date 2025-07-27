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
python3 multi_sample_evaluation.py \
  data/esqualo_2022_fall_original.csv \
  --field material \
  --brand esqualo \
  --output-dir evaluation_results \
  --run all \
  --num-samples 10
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
    "material": "Material",
    "color_name": "Color"
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
