# Viewing and Interpreting Results

This guide explains how to view and interpret the results from data quality detection runs.

## Result Files Overview

After running detection, you'll find several output files in your output directory:

- `report.json` - Comprehensive detection results
- `viewer_report.json` - Formatted for the web viewer
- `anomaly_summary.csv` - Summary of all detected anomalies
- `sample_with_errors.csv` - Data with injected errors (if using evaluation mode)
- `sample_with_results.csv` - Original data with detection results
- `confusion_matrix/` - Performance visualization images

## Using the HTML Viewer

The interactive HTML viewer is the easiest way to explore results:

1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in a modern web browser
2. Upload the generated CSV and JSON files from your output directory:
   - Upload the CSV file (anomaly_summary.csv or sample_with_results.csv)
   - Upload the JSON report (viewer_report.json)
3. Use the interface to:
   - Filter by detection confidence
   - Sort by different criteria
   - View detailed explanations
   - Export filtered results

### Viewer Features

- **Confidence Filtering**: Show only high/medium/low confidence detections
- **Field Selection**: Focus on specific fields
- **Detection Method Filter**: View results from specific methods
- **Search**: Find specific records or values
- **Export**: Download filtered results as CSV

## Understanding the CSV Output

The anomaly_summary.csv contains detected anomalies with columns for:

- `row_index` - Row number in original data
- `column_name` - Field where anomaly was detected
- `detection_method` - Which method found the anomaly
- `error_type` - Type of issue detected
- `confidence` - Detection confidence (0-1)
- `details` - Additional information
- `error_data` - The problematic value

### Interpreting Confidence Scores

- **0.9-1.0**: Very high confidence detection
- **0.7-0.9**: High confidence
- **0.5-0.7**: Medium confidence
- **0.3-0.5**: Low confidence
- **0.0-0.3**: Very low confidence

## Reading the Detection Report

The report.json file contains comprehensive results including:

- Detection summary statistics
- Performance metrics (if evaluation mode)
- Field-by-field breakdown
- Method-specific results
- Confusion matrix data (if applicable)

Key sections to review:
- `summary`: Overall detection statistics
- `metrics`: Performance metrics like precision, recall, F1
- `by_field`: Results broken down by field
- `by_method`: Results broken down by detection method

## Analyzing Results by Field

Look for patterns in the field-level results:

1. **High Detection Rates**: Fields with >10% detection rate may have systemic issues
2. **Low Confidence Detections**: May indicate need for threshold adjustment
3. **Method Agreement**: Records flagged by multiple methods are likely true positives

## Analyzing Results by Method

Compare detection methods to understand their effectiveness:

- **Validation**: Highest precision, rule-based detections
- **Pattern-Based**: Good for structured fields, medium precision
- **ML-Based**: Semantic understanding, may need tuning
- **LLM-Based**: Most sophisticated but resource-intensive

## Common Patterns to Look For

### False Positives

Signs of potential false positives:
- Very low confidence scores (&lt;0.3)
- Only flagged by one method
- Common values being flagged
- Inconsistent detection patterns

### True Positives

Signs of likely true positives:
- High confidence scores (&gt;0.8)
- Flagged by multiple methods
- Clear rule violations
- Consistent patterns across similar records

## Exporting and Sharing Results

### For Stakeholders

Create a summary report:

```bash
# Generate executive summary
python -c "
import json
with open('report.json', 'r') as f:
    report = json.load(f)
    summary = report.get('summary', {})
    print(f\"Total Anomalies: {summary.get('total_anomalies', 0)}\")
    if 'metrics' in report:
        metrics = report['metrics']
        print(f\"Precision: {metrics.get('precision', 0):.2f}\")
        print(f\"Recall: {metrics.get('recall', 0):.2f}\")
"
```

### For Data Teams

Export detailed analysis:

```python
import pandas as pd

# Load anomaly summary
df = pd.read_csv('anomaly_summary.csv')

# Filter high-confidence issues
high_conf = df[df['confidence'] > 0.8]

# Group by field and error type
issues_by_field = high_conf.groupby(['column_name', 'error_type']).size()

# Export for further analysis
issues_by_field.to_csv('high_confidence_issues.csv')
```

## Taking Action on Results

### Immediate Actions

1. **Critical Errors** (confidence > 0.9): Fix immediately
2. **Validation Failures**: Update source data or rules
3. **Pattern Anomalies**: Investigate root cause

### Process Improvements

1. **Recurring Issues**: Update data entry processes
2. **False Positives**: Adjust detection thresholds
3. **New Patterns**: Add validation rules

## Next Steps

- [Evaluating Detection Performance](evaluating-performance.md)
- [Optimizing Detection Weights](optimizing-weights.md)
- [Adding New Fields](../development/new-fields.md)