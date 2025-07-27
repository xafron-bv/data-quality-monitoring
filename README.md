# Data Quality Monitoring System

A comprehensive data quality monitoring system that combines rule-based validation, pattern-based anomaly detection, and ML-based anomaly detection using a unified interface.

## 📚 Documentation Index

All detailed documentation is now located in the [`docs/`](./docs/) directory:

- [System Class Hierarchy](./docs/CLASS_HIERARCHY_DOCUMENTATION.md)
- [ML Anomaly Detection Documentation](./docs/ML_Anomaly_Detection_Documentation.md)
- [ML Anomaly Detection Overview](./docs/ML_Anomaly_Detection_Overview.md)
- [Demo Commands](./docs/demo_commands.md)
- [Weighted Combination Detection](./docs/WEIGHTED_COMBINATION.md)
- Prompts Documentation:
  - [Rule Creation Guide](./docs/prompts/rule-create.md)
  - [Validator Creation Guide](./docs/prompts/validator-create.md)
  - [Validator Edit Guide](./docs/prompts/validator-edit.md)

## 🏗️ System Architecture

The system provides three complementary approaches to data quality monitoring:

1. **Validation** - Rule-based validation with high confidence detection
2. **Anomaly Detection** - Pattern-based anomaly detection for medium confidence issues  
3. **ML Detection** - Machine learning-based anomaly detection using sentence transformers
4. **LLM Detection** - Language model-based anomaly detection using fine-tuned transformers

## 🎯 Detection Combination Methods

The system supports two approaches for combining detection results:

- **Priority-Based** (default): Uses fixed hierarchy (validation > pattern > ML > LLM)
- **Weighted Combination** (new): Uses performance-based weights for each field/method combination

See [Weighted Combination Documentation](./docs/WEIGHTED_COMBINATION.md) for detailed usage.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Brand Configuration](#brand-configuration)
- [ML Model Training](#ml-model-training)
- [Running Evaluations](#running-evaluations)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Performance Analysis](#performance-analysis)

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r anomaly_detectors/ml_based/requirements.txt
## Optionally if you want CUDA, reinstall torch with CUDA support (look up the command from https://pytorch.org/get-started/locally/)
```

### Basic Workflow

1. **Train ML Models** (optional but recommended)
2. **Run Evaluations** to test all detection methods
3. **Analyze Results** from generated reports

### Weighted Combination Quick Start

For improved detection accuracy, use the weighted combination approach:

```bash
# 1. Run evaluation to generate performance data
python3 demo.py --brand your_brand --injection-intensity 0.15 --core-fields-only --enable-validation --enable-pattern --enable-ml --enable-llm

# 2. Generate detection weights from performance results
python3 generate_detection_weights.py --input-file demo_results/demo_analysis_unified_report.json --output-file detection_weights.json --verbose

# 3. Run detection with weighted combination
python3 demo.py --brand your_brand --core-fields-only --enable-validation --enable-pattern --enable-ml --enable-llm --use-weighted-combination --weights-file detection_weights.json
```

See [Weighted Combination Documentation](./docs/WEIGHTED_COMBINATION.md) for complete details.

## 🏢 Brand Configuration

The system supports multiple brands with different column mappings and data sources. Each brand has its own JSON configuration file in the `brand_configs/` directory.

### Managing Brands

Use the `manage_brands.py` utility to manage brand configurations:

```bash
# List all configured brands
python manage_brands.py --list

# Create a new brand configuration template
python manage_brands.py --create new_brand_name

# Show detailed configuration for a brand
python manage_brands.py --show your_brand

# Validate a brand configuration
python manage_brands.py --validate new_brand_name
```

### Brand Configuration Format

Each brand configuration file contains:
- **field_mappings**: Maps standard field names to brand-specific column names
- **default_data_path**: Path to the brand's main data file
- **training_data_path**: Path to the brand's training data
- **ml_models_path**: Path where ML models for this brand are stored
- **enabled_fields**: List of fields to analyze for this brand
- **custom_thresholds**: Brand-specific detection thresholds

**Important**: Validation and anomaly detection rules are global and shared across all brands. The same rules work for all brands - only the column names are mapped differently.

Example configuration:
```json
{
  "brand_name": "mybrand",
  "field_mappings": {
    "material": "ProductComposition",
    "color_name": "ColorDescription",
    "size": "SizeName"
  },
  "default_data_path": "data/mybrand_data.csv",
  "training_data_path": "data/mybrand_training.csv",
  "ml_models_path": "anomaly_detectors/ml_based/results/mybrand",
  "enabled_fields": ["material", "color_name", "size"]
}
```

### Using Different Brands

Specify the brand when running evaluations or demos:

```bash
# Run evaluation for a specific brand
python evaluate.py data/mybrand_data.csv --field material --brand mybrand

# Run demo with brand configuration
python demo.py --brand mybrand

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
python evaluate.py data/your_data.csv \
  --brand your_brand \
  --field="material" \
  --validator="material" \
  --ml-detector \
  --run="all" \
  --output-dir="evaluation_results/material_evaluation"
```

#### Evaluate with Validation + ML Detection Only
```bash
python evaluate.py data/your_data.csv \
  --brand your_brand \
  --field="color_name" \
  --validator="color_name" \
  --ml-detector \
  --run="validation" \
  --output-dir="evaluation_results/color_validation_ml"
```

#### Evaluate ML Detection Only
```bash
python evaluate.py data/your_data.csv \
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
├── evaluate.py                     # Main evaluation script
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
  }
}
```

### Evaluation Configuration

Modify evaluation scripts to test different combinations:

```bash
# In run_evaluations.sh
EVALUATIONS=(
  "material:material"
  "color_name:colour_name" 
  "category:article_structure_name_2"
)
```

### Field-Column Mapping

Configure field mappings in `anomaly_detectors/ml_based/field_column_map.py`:

```python
def get_field_to_column_map():
    return {
        "material": "material",
        "color_name": "colour_name",
        "category": "article_structure_name_2"
    }
```

## 🚀 Advanced Usage

### Custom Threshold Testing

Test ML detection with different sensitivity levels:

```bash
cd anomaly_detectors/ml_based

# High sensitivity (catch subtle anomalies)
python index.py ../../data/test_data.csv --check-anomalies material --threshold 0.95

# Balanced detection
python index.py ../../data/test_data.csv --check-anomalies material --threshold 0.6

# Conservative detection (only obvious anomalies)
python index.py ../../data/test_data.csv --check-anomalies material --threshold 0.3
```

### Programmatic Usage

```python
from unified_detection_interface import CombinedDetector, UnifiedReporter
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
import pandas as pd

# Load data
df = pd.read_csv("data/test_data.csv")

# Initialize ML detector
ml_detector = MLAnomalyDetector(field_name="material", threshold=0.6)
ml_detector.learn_patterns(df, "material")

# Create unified detector
combined_detector = CombinedDetector(ml_detector=ml_detector)

# Run detection
results = combined_detector.detect_issues(
    df, 
    "material",
    enable_ml_detection=True,
    ml_threshold=0.6
)

# Generate report
reporter = UnifiedReporter(include_technical_details=True)
report = reporter.generate_report(results, df)
```

### Training Custom Models

Train models for new data types:

1. **Create error injection rules** in `error_injection_rules/new_field.json`
2. **Add field mapping** in `anomaly_detectors/ml_based/field_column_map.py`
3. **Train the model**:
   ```bash
   cd anomaly_detectors/ml_based
   python index.py ../../data/your_data.csv --rules new_field --use-hp-search
   ```

## 📊 Performance Analysis

### Understanding Metrics

- **Recall**: Percentage of actual anomalies detected (higher = fewer missed issues)
- **Precision**: Percentage of detections that are actual anomalies (higher = fewer false alarms)
- **F1 Score**: Balanced measure combining recall and precision

### Performance Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.9+ | Excellent performance |
| 0.7-0.9 | Good performance |
| 0.5-0.7 | Moderate performance |
| <0.5 | Needs improvement |

### ML Model Performance Expectations

Based on hyperparameter optimization results:

| Field | Expected Recall | Expected Precision | Notes |
|-------|----------------|-------------------|--------|
| `color_name` | 1.0 | 0.85+ | Excellent performance |
| `category` | 1.0 | 0.90+ | Excellent performance |
| `material` | 0.88+ | 0.65+ | Good performance |
| `ean` | 0.5+ | 0.30+ | Moderate performance |

### Troubleshooting Performance Issues

**Low Recall (Missing Anomalies)**:
- Lower ML detection threshold (e.g., 0.95 → 0.6)
- Retrain with more diverse error injection rules
- Use hyperparameter search to find better parameters

**Low Precision (Too Many False Positives)**:
- Raise ML detection threshold (e.g., 0.6 → 0.3)
- Add more training data
- Review and refine error injection rules

**Training Failures**:
- Check that error injection rules exist for the field
- Ensure sufficient clean training data (>100 unique values recommended)
- Verify field-column mapping is correct

## 🔧 Tips and Best Practices

### Training Best Practices

1. **Start with hyperparameter search** for new fields to find optimal settings
2. **Use adequate training data** - at least 100+ unique values per field
3. **Create realistic error injection rules** that match actual data quality issues
4. **Train on clean data** to establish good baselines

### Evaluation Best Practices

1. **Test all three approaches** (validation + anomaly + ML) for comprehensive coverage
2. **Use sufficient sample sizes** (32+ samples) for reliable metrics
3. **Review false positives** to improve detection rules
4. **Analyze false negatives** to identify missing error patterns

### Production Usage Tips

1. **Start with conservative thresholds** to minimize disruption
2. **Monitor detection rates** and adjust thresholds based on feedback
3. **Combine approaches** - use validation for high-confidence issues, ML for subtle anomalies
4. **Regularly retrain models** as data patterns evolve

---

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Review the generated summary reports
3. Examine the JSON results for detailed metrics
4. Verify your field-column mappings and error injection rules

## 🔄 Updates and Maintenance

### Updating Models
- Retrain models periodically with new data
- Monitor performance metrics over time
- Update error injection rules based on real-world findings

### System Updates
- Keep dependencies updated
- Review and tune thresholds based on operational experience
- Expand error injection rules as new patterns are discovered
