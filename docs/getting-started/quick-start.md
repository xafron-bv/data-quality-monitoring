# Quick Start Guide

Welcome to the Data Quality Monitoring System! This guide will help you run your first detection in minutes.

## Prerequisites

Before starting, ensure you have completed the [Installation Guide](installation.md).

## Your First Detection Run

The easiest way to start is with the single sample demo:

```python
# run_demo.py
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "data/sample_data.csv",
    "--output-dir", "results/quick_start",
    "--enable-validation", "--enable-pattern", "--enable-ml",
])
```

This will:
1. Load your data
2. Inject synthetic anomalies for testing
3. Run enabled detection methods
4. Generate reports and visualizations

## Understanding the Output

### Generated Files
Check your output directory. You should see files like:
- `<sample>_viewer_report.json` (for the HTML viewer)
- `<sample>_unified_report.json` (for metrics and weights generation)
- `<sample>_overall_confusion_matrix.png`
- `<sample>_per_field_confusion_matrix.png`
- `<sample>_detection_type_confusion_matrix.png`
- `<sample>_performance_comparison.png`
- `<sample>_summary_visualization.png`

Note: `<sample>` defaults to names used by the demo, e.g. `demo_analysis` or `confusion_matrix_analysis`.

### Web Viewer
1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. Upload the generated CSV sample (e.g., `results/quick_start/demo_sample.csv`)
3. Upload the JSON file `<sample>_viewer_report.json` from your output directory
4. Explore interactive visualizations

## Try Different Detection Methods

Now that you've seen the basic demo, try running with specific detection methods by adjusting the flags you pass in Python:

```python
import subprocess
subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "data/sample_data.csv",
    "--enable-validation",
])

subprocess.run([
    "python", "main.py", "single-demo",
    "--data-file", "data/sample_data.csv",
    "--enable-pattern",
])
```

## What's Next?

Now that you've run your first detection:
- Learn more workflows in the [Basic Usage Guide](basic-usage.md)
- Understand your data with the [Data Analysis Guide](../user-guides/analyzing-results.md)
- Configure detection for your needs in the [Configuration Reference](../reference/configuration.md)