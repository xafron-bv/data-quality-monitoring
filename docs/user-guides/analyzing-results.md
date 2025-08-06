# Analyzing Results

This guide covers how to analyze your data before detection and interpret the results afterwards.

## Before Detection: Understanding Your Data

### Explore Data Structure

Before running detection, understand your data:

```bash
# View the first few rows
head -n 10 your_data.csv

# Count records
wc -l your_data.csv
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

After running detection, you'll find in your output directory:

- `report.json` - Comprehensive detection results
- `viewer_report.json` - Formatted for the web viewer
- `anomaly_summary.csv` - Summary of all detected anomalies
- `sample_with_errors.csv` - Data with injected errors (if using evaluation mode)
- `sample_with_results.csv` - Original data with detection results
- `confusion_matrix/` - Performance visualization images

### Using the HTML Viewer

The interactive viewer is the easiest way to explore results:

1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in your browser
2. Upload files from your output directory:
   - CSV file (anomaly_summary.csv or sample_with_results.csv)
   - JSON report (viewer_report.json)
3. Use the interface to:
   - Filter by confidence level
   - Sort by different criteria
   - View detailed explanations
   - Export filtered results

### Understanding the CSV Output

The anomaly_summary.csv contains:

- `row_index` - Row number in original data
- `column_name` - Field where anomaly was detected
- `detection_method` - Which method found the anomaly
- `error_type` - Type of issue detected
- `confidence` - Detection confidence (0-1)
- `details` - Additional information
- `error_data` - The problematic value

### Interpreting Confidence Scores

- **0.8-1.0**: High confidence - likely real issues
- **0.5-0.8**: Medium confidence - review recommended
- **0.0-0.5**: Low confidence - possible false positives

### Reading the JSON Report

The report.json contains:
- Detection summary statistics
- Performance metrics (if evaluation mode)
- Field-by-field breakdown
- Method-specific results

Key sections:
- `summary`: Overall detection statistics
- `field_results`: Results per field
- `metrics`: Performance metrics (precision, recall, F1)

## Analysis Workflow

### 1. Quick Overview

```python
import json
import pandas as pd

# Load results
with open('report.json', 'r') as f:
    report = json.load(f)

# Check summary
print(f"Total anomalies: {report['summary'].get('total_anomalies', 0)}")
if 'metrics' in report:
    print(f"Precision: {report['metrics'].get('precision', 0):.2f}")
    print(f"Recall: {report['metrics'].get('recall', 0):.2f}")
```

### 2. Deep Dive Analysis

```python
# Load anomaly details
df = pd.read_csv('anomaly_summary.csv')

# Analyze by field
field_counts = df['column_name'].value_counts()
print("Issues by field:")
print(field_counts)

# High confidence issues
high_conf = df[df['confidence'] > 0.8]
print(f"\nHigh confidence issues: {len(high_conf)}")
```

### 3. Pattern Analysis

Group issues to find patterns:

```python
# Group by error type
error_patterns = df.groupby(['column_name', 'error_type']).size()
print("\nError patterns:")
print(error_patterns.sort_values(ascending=False).head(10))
```

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
- Check that detection methods were enabled
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