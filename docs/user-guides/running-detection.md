# Running Detection

This guide covers how to run data quality detection on your datasets using various methods and configurations.

## Basic Detection

The simplest way to run detection is using the demo command:

```bash
python main.py single-demo --data-file your_data.csv
```

This runs detection with default settings on your entire dataset.

## Configuring Detection Methods

You can enable specific detection methods:

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --enable-validation \    # Rule-based validation
    --enable-pattern \       # Pattern-based anomaly detection
    --enable-ml \           # Machine learning detection
    --enable-llm            # Language model detection
```

## Setting Detection Thresholds

Adjust sensitivity for each detection method:

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --validation-threshold 0.0 \    # Most strict (0.0)
    --anomaly-threshold 0.7 \       # Medium confidence
    --ml-threshold 0.7 \            # Medium confidence
    --llm-threshold 0.6             # Slightly more lenient
```

### Threshold Guidelines

- **0.0**: Highest confidence, fewest false positives
- **0.5**: Balanced detection
- **0.8**: More sensitive, may have more false positives
- **1.0**: Detect everything (not recommended)

## Working with Large Datasets

For large datasets, use these strategies:

### 1. Sample Processing

To process a sample of your data, first create a subset:

```bash
# Create a sample file
head -n 1000 large_data.csv > sample_data.csv

# Run detection on the sample
python main.py single-demo \
    --data-file sample_data.csv
```

### 2. Core Fields Only

```bash
python main.py single-demo \
    --data-file large_data.csv \
    --core-fields-only
```

### 3. Specific Fields

Note: The single-demo command processes all configured fields. To process specific fields only, you can modify your brand configuration or use the multi-eval command for field-specific evaluation.

## Output Options

### Specify Output Directory

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --output-dir results/2024-01-detection
```

### Output Files

The single-demo command automatically generates:
- JSON reports (report.json, viewer_report.json)
- CSV summaries (anomaly_summary.csv)
- Result files with detection information
- Confusion matrix visualizations (if evaluation mode)

## Using Weighted Combination

For optimized detection based on historical performance:

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --use-weighted-combination \
    --weights-file detection_weights.json
```

## Injection Testing

Test detection performance with synthetic errors:

```bash
python main.py single-demo \
    --data-file clean_data.csv \
    --injection-intensity 0.2    # Inject errors in 20% of data
```

Note: Error injection is randomized. The exact errors will vary between runs.

## Viewing Results

After detection completes:

1. **HTML Viewer**: Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. **CSV Results**: Review the generated CSV files in your output directory
3. **JSON Report**: Detailed metrics in the report.json file

## Example Workflows

### Quick Quality Check

```bash
# Fast check with pattern detection only
python main.py single-demo \
    --data-file daily_upload.csv \
    --enable-pattern
```

### Full Production Run

```bash
# Comprehensive detection with all methods
python main.py single-demo \
    --data-file production_data.csv \
    --enable-validation \
    --enable-pattern \
    --enable-ml \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.6 \
    --ml-threshold 0.7 \
    --output-dir results/production_$(date +%Y%m%d)
```

### Testing New Configuration

```bash
# Test with injection to validate configuration
python main.py single-demo \
    --data-file test_data.csv \
    --injection-intensity 0.3 \
    --enable-validation \
    --enable-pattern
```

## Troubleshooting

### Out of Memory Errors

- Use `--core-fields-only`
- Create a smaller sample file first
- Disable ML/LLM detection
- Process in batches

### Slow Performance

- Enable GPU if available
- Use `--enable-pattern` only for quick checks
- Use `--core-fields-only` to process fewer fields

### No Detections Found

- Check field mappings in brand configuration
- Lower detection thresholds
- Verify data format matches expectations