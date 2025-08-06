# Quick Start Guide

Welcome to the Data Quality Monitoring System! This guide will help you run your first detection in minutes.

## Prerequisites

Before starting, ensure you have completed the [Installation Guide](installation.md).

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
1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. Upload the generated CSV and JSON files
3. Explore interactive visualizations

## Try Different Detection Methods

Now that you've seen the basic demo, try running with specific detection methods:

```bash
# Fast validation only
python main.py single-demo \
    --data-file data/sample_data.csv \
    --enable-validation

# Or try pattern-based detection
python main.py single-demo \
    --data-file data/sample_data.csv \
    --enable-pattern
```

## What's Next?

Now that you've run your first detection:
- Learn more workflows in the [Basic Usage Guide](basic-usage.md)
- Understand your data with the [Data Analysis Guide](../user-guides/analyzing-results.md)
- Configure detection for your needs in the [Configuration Reference](../reference/configuration.md)