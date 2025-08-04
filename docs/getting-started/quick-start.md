# Quick Start Guide

Welcome to the Data Quality Monitoring System! This guide will help you get started in minutes.

## Prerequisites

- Python 3.8+
- pip or conda
- 4GB+ RAM recommended

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-quality-monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python main.py --help
```

## Your First Detection Run

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

## Understanding the Output

### Console Output
You'll see real-time progress:
```
🔍 Processing sample: quick_start_sample
📊 Total fields to check: 15
━━━━━━━━━━━━━━━━━━━━ 100% 15/15
✅ Detection complete!
```

### Generated Files
Check your output directory:
- `report.json` - Detailed detection results
- `viewer_report.json` - Formatted for the web viewer
- `anomaly_summary.csv` - Summary of all detections
- `confusion_matrix/` - Performance visualizations

### Web Viewer
1. Open `data_quality_viewer.html` in your browser
2. Upload the generated CSV and JSON files
3. Explore interactive visualizations

## Basic Detection Options

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
    --enable-ml
```

### Adjust Detection Sensitivity

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.7 \
    --ml-threshold 0.8
```

### Control Anomaly Injection

```bash
# No injection (test on real data)
python main.py single-demo \
    --data-file your_data.csv \
    --injection-intensity 0.0

# Heavy injection for stress testing
python main.py single-demo \
    --data-file your_data.csv \
    --injection-intensity 0.5 \
    --max-issues-per-row 3
```

## Common Use Cases

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

## Next Steps

1. **Analyze Your Data**: Use `analyze-column` to understand your fields
2. **Train Models**: Train ML models for better detection
3. **Configure Rules**: Customize detection rules for your data
4. **Batch Processing**: Use `multi-eval` for large-scale evaluation

See the [Basic Usage Guide](basic-usage.md) for more detailed examples.

## Troubleshooting

If you encounter issues:
- Check that all requirements are installed
- Verify your CSV file format matches the expected structure
- Ensure brand configuration is properly set up
- Check the logs for detailed error messages

---

Ready to dive deeper? Check out our guides on:
- [Understanding the different entrypoints](understanding-entrypoints.md)
- [Configuring detection methods](../configuration/brand-config.md)
- [Running performance evaluations](../getting-started/basic-usage.md)