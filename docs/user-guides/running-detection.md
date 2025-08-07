# Running Detection

This guide covers how to run data quality detection on your datasets using various methods and configurations.

## Basic Detection

The simplest way to run detection is using the demo command:

```python
import subprocess
subprocess.run(["python", "main.py", "single-demo", "--data-file", "your_data.csv"]) 
```

This runs detection with default thresholds. Methods are disabled unless enabled with flags.

## Configuring Detection Methods

Enable specific detection methods:

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "your_data.csv",
    "--enable-validation",
    "--enable-pattern",
    "--enable-ml",
    # "--enable-llm",
])
```

## Setting Detection Thresholds

Adjust sensitivity for each detection method:

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "your_data.csv",
    "--validation-threshold", "0.0",
    "--anomaly-threshold", "0.7",
    "--ml-threshold", "0.7",
    "--llm-threshold", "0.6",
])
```

### Threshold Guidelines

- 0.0: Highest confidence, fewest false positives (validation)
- 0.5: Balanced detection
- 0.8: More sensitive
- 1.0: Detect nothing

## Working with Large Datasets

### 1. Sample Processing

```python
import pandas as pd
# Create a sample file of first 1000 rows
pd.read_csv("large_data.csv").head(1000).to_csv("sample_data.csv", index=False)

import subprocess
subprocess.run(["python", "main.py", "single-demo", "--data-file", "sample_data.csv"]) 
```

### 2. Core Fields Only

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "large_data.csv",
    "--core-fields-only",
])
```

### 3. Specific Fields

Note: The single-demo command processes all configured fields. To process specific fields only, adjust your brand configuration or use the multi-eval command for field-specific evaluation.

## Output Options

### Specify Output Directory

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "your_data.csv",
    "--output-dir", "results/2024-01-detection",
])
```

### Output Files

The single-demo command generates:
- `<sample>_viewer_report.json`
- `<sample>_unified_report.json`
- Confusion matrix visualizations as PNGs in the output directory

## Using Weighted Combination

For optimized detection based on historical performance:

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "your_data.csv",
    "--use-weighted-combination",
    "--weights-file", "detection_weights.json",
])
```

## Injection Testing

Test detection performance with synthetic errors:

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "clean_data.csv",
    "--injection-intensity", "0.2",
])
```

Note: Error injection is randomized. The exact errors will vary between runs.

## Viewing Results

After detection completes:

1. HTML Viewer: Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. Upload the generated sample CSV (e.g., `demo_sample.csv`) and `<sample>_viewer_report.json`
3. Review metrics in `<sample>_unified_report.json`

## Example Workflows

### Quick Quality Check

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "daily_upload.csv",
    "--enable-pattern",
])
```

### Full Production Run

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "production_data.csv",
    "--enable-validation", "--enable-pattern", "--enable-ml",
    "--validation-threshold", "0.0",
    "--anomaly-threshold", "0.6",
    "--ml-threshold", "0.7",
    "--output-dir", "results/production_20240101",
])
```

### Testing New Configuration

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "test_data.csv",
    "--injection-intensity", "0.3",
    "--enable-validation", "--enable-pattern",
])
```

## Troubleshooting

### Out of Memory Errors
- Use `--core-fields-only`
- Create a smaller sample first
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