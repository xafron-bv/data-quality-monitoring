# Viewing and Interpreting Results

This guide explains how to view and interpret the results from data quality detection runs.

## Result Files Overview

After running detection, you'll find several output files:

- `data_quality_viewer.html` - Interactive web viewer
- `*_predictions.csv` - Detailed predictions for each record
- `detection_report.json` - Summary metrics and statistics
- `*_detailed_report.json` - Field-by-field analysis (if enabled)

## Using the HTML Viewer

The interactive HTML viewer is the easiest way to explore results:

1. Open `data_quality_viewer.html` in a modern web browser
2. Upload the generated CSV and JSON files when prompted
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

## Understanding the Predictions CSV

The predictions CSV contains one row per record with columns for:

- Original data fields
- `detection_result` - Combined detection outcome
- `confidence_score` - Overall confidence (0-1)
- `detection_methods` - Which methods flagged the record
- Method-specific scores and explanations

### Interpreting Confidence Scores

- **0.9-1.0**: Very high confidence detection
- **0.7-0.9**: High confidence
- **0.5-0.7**: Medium confidence
- **0.3-0.5**: Low confidence
- **0.0-0.3**: Very low confidence

## Reading the Detection Report

The JSON report provides summary statistics:

```json
{
  "summary": {
    "total_records": 1000,
    "total_detections": 150,
    "detection_rate": 0.15,
    "average_confidence": 0.72
  },
  "by_field": {
    "material": {
      "detections": 45,
      "detection_rate": 0.045,
      "top_issues": ["Invalid material code", "Unusual pattern"]
    }
  },
  "by_method": {
    "validation": {
      "detections": 80,
      "precision": 0.95,
      "average_confidence": 0.98
    }
  }
}
```

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
with open('detection_report.json', 'r') as f:
    report = json.load(f)
    print(f\"Detection Rate: {report['summary']['detection_rate']:.1%}\")
    print(f\"High Confidence Issues: {report['summary']['high_confidence_count']}\")
"
```

### For Data Teams

Export detailed analysis:

```python
import pandas as pd

# Load predictions
df = pd.read_csv('predictions.csv')

# Filter high-confidence issues
high_conf = df[df['confidence_score'] > 0.8]

# Group by field and issue type
issues_by_field = high_conf.groupby(['field', 'detection_reason']).size()

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