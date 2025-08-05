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

```bash
python main.py single-demo \
    --data-file large_data.csv \
    --sample-size 1000 \
    --random-seed 42
```

### 2. Core Fields Only

```bash
python main.py single-demo \
    --data-file large_data.csv \
    --core-fields-only
```

### 3. Specific Fields

```bash
python main.py single-demo \
    --data-file large_data.csv \
    --fields material color_name category
```

## Output Options

### Specify Output Directory

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --output-dir results/2024-01-detection
```

### Control Output Formats

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --save-predictions \     # Save detailed predictions
    --no-html               # Skip HTML viewer generation
```

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
    --injection-intensity 0.2 \    # Inject errors in 20% of data
    --injection-seed 42
```

## Viewing Results

After detection completes:

1. **HTML Viewer**: Open `data_quality_viewer.html` in your browser
2. **CSV Results**: Import `*_predictions.csv` into Excel or similar
3. **JSON Report**: Detailed metrics in `detection_report.json`

## Example Workflows

### Quick Quality Check

```bash
# Fast check with pattern detection only
python main.py single-demo \
    --data-file daily_upload.csv \
    --enable-pattern \
    --sample-size 500
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
    --enable-pattern \
    --sample-size 100
```

## Troubleshooting

### Out of Memory Errors

- Use `--core-fields-only`
- Reduce `--sample-size`
- Disable ML/LLM detection
- Process in batches

### Slow Performance

- Enable GPU if available
- Use `--enable-pattern` only for quick checks
- Reduce number of fields with `--fields`

### No Detections Found

- Check field mappings in brand configuration
- Lower detection thresholds
- Verify data format matches expectations

## Next Steps

- [Evaluating Detection Performance](evaluating-performance.md)
- [Optimizing Detection Weights](optimizing-weights.md)
- [Viewing and Interpreting Results](viewing-results.md)