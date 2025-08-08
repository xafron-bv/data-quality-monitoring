# Analyzing Results

This guide covers how to analyze your data before detection and interpret the results afterwards.

## Before Detection: Understanding Your Data

### Explore Data Structure

Use Python to understand your data:

```python
import pandas as pd

df = pd.read_csv("your_data.csv")
print(df.head(10))
print(len(df))  # row count
```

### Analyze Individual Columns

Use the `analyze-column` command to understand field characteristics:

```bash
python main.py analyze-column your_data.csv column_name
```

This shows:
- Unique value distribution
- Common patterns
- Potential anomalies
- Recommended detection methods

### Identify Key Fields

Focus on fields that are:
- Critical for business operations
- Prone to quality issues
- Used in downstream processes

## After Detection: Interpreting Results

### Result Files Overview

After running detection, your output directory includes:
- `<sample>_viewer_report.json` (for the HTML viewer)
- `<sample>_unified_report.json` (summary and metrics)
- PNG visualizations (confusion matrices and summaries)

### Using the HTML Viewer

The interactive viewer is the easiest way to explore results:

1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. Upload files from your output directory:
   - CSV sample (e.g., `demo_sample.csv`)
   - `<sample>_viewer_report.json`
3. Use the interface to:
   - Filter by confidence level
   - Sort by different criteria
   - View detailed explanations
   - Export filtered results

### Interpreting Confidence Scores

- 0.8-1.0: High confidence - likely real issues
- 0.5-0.8: Medium confidence - review recommended
- 0.0-0.5: Low confidence - possible false positives

### Reading the Unified Report

The `<sample>_unified_report.json` contains:
- Field-by-field performance metrics
- Precision/recall/F1 summaries
- Counts per detection method

Quick peek:
```python
import json
with open("results/demo_analysis_unified_report.json") as f:
    report = json.load(f)
print(report["fields"].keys())  # fields with metrics
```

## Analysis Workflow

### 1. Quick Overview

```python
import json
with open('results/demo_analysis_unified_report.json', 'r') as f:
    report = json.load(f)
summary = report.get('fields', {})
print(f"Fields analyzed: {len(summary)}")
```

### 2. Deep Dive Analysis

```python
# Inspect a single field's metrics
data = summary.get('material', {})
print(data)
```

### 3. Pattern Analysis

Group issues using the viewer CSV filtering and the viewer report categories to find patterns by error type and field.

## Best Practices

### For Data Analysis
- Start with critical business fields
- Use analyze-column before full detection
- Document expected patterns

### For Result Interpretation
- Focus on high-confidence detections first
- Look for patterns across multiple records
- Consider business context when reviewing
- Export and share findings with stakeholders

## Troubleshooting

### No Results Generated
- Ensure detection methods were enabled
- Verify data file format is correct
- Review console output for errors

### Too Many False Positives
- Increase detection thresholds
- Review and update validation rules
- Consider training custom ML models

### Missing Expected Issues
- Lower detection thresholds
- Enable additional detection methods
- Check field mappings in brand configuration