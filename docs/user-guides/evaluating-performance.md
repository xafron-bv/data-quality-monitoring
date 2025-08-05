# Evaluating Detection Performance

This guide explains how to evaluate and measure the performance of your data quality detection system.

## Understanding Performance Metrics

The system tracks several key metrics:

- **Precision**: How many detections were correct (true positives / all positives)
- **Recall**: How many errors were caught (true positives / all actual errors)  
- **F1 Score**: Harmonic mean of precision and recall
- **Detection Rate**: Percentage of records flagged
- **Confidence Distribution**: Spread of confidence scores

## Running Evaluation

### Basic Evaluation

Use the multi-eval command for systematic evaluation:

```bash
python main.py multi-eval \
    --input your_data.csv \
    --sample-size 1000 \
    --output-dir evaluation_results
```

### Evaluation with Known Errors

If you have labeled data with known errors:

```bash
python main.py multi-eval \
    --input labeled_data.csv \
    --ground-truth-column is_error \
    --sample-size 1000
```

### Synthetic Error Evaluation

Test with injected errors to measure performance:

```bash
python main.py multi-eval \
    --input clean_data.csv \
    --injection-intensity 0.2 \
    --injection-seed 42 \
    --sample-size 1000
```

## Interpreting Evaluation Results

### Confusion Matrix

The evaluation generates a confusion matrix showing:

```
                 Predicted
              Error   No Error
Actual Error   TP      FN
   No Error    FP      TN

TP = True Positives (correctly detected errors)
FN = False Negatives (missed errors)
FP = False Positives (false alarms)
TN = True Negatives (correctly identified clean data)
```

### Performance by Field

Review field-specific performance:

```json
{
  "material": {
    "precision": 0.92,
    "recall": 0.85,
    "f1_score": 0.88,
    "support": 200
  },
  "color_name": {
    "precision": 0.78,
    "recall": 0.90,
    "f1_score": 0.84,
    "support": 150
  }
}
```

### Performance by Method

Compare detection methods:

```json
{
  "validation": {
    "precision": 0.98,
    "recall": 0.75,
    "f1_score": 0.85
  },
  "pattern_based": {
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85
  }
}
```

## Monitoring Over Time

### Set Up Regular Evaluation

Create a monitoring script:

```bash
#!/bin/bash
# monitor_detection.sh

DATE=$(date +%Y%m%d)
python main.py multi-eval \
    --input daily_data.csv \
    --sample-size 500 \
    --output-dir monitoring/$DATE \
    --save-detailed-report

# Track metrics over time
python analyze_trends.py monitoring/
```

### Key Metrics to Track

1. **Precision Trends**: Are we maintaining low false positive rates?
2. **Recall Trends**: Are we catching most errors?
3. **Detection Rate Changes**: Is the error rate changing?
4. **Method Performance**: Which methods are most effective?

## Benchmarking

### Create Benchmark Dataset

1. Select representative samples
2. Manually label errors
3. Include edge cases
4. Document labeling criteria

```bash
# Run benchmark evaluation
python main.py multi-eval \
    --input benchmark_data.csv \
    --ground-truth-column manually_labeled \
    --output-dir benchmark_results \
    --save-predictions
```

### Compare Configurations

Test different threshold settings:

```bash
# Conservative configuration
python main.py multi-eval \
    --input benchmark_data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.8 \
    --output-dir results/conservative

# Aggressive configuration  
python main.py multi-eval \
    --input benchmark_data.csv \
    --validation-threshold 0.3 \
    --anomaly-threshold 0.5 \
    --output-dir results/aggressive
```

## Analyzing False Positives

### Common Causes

1. **Threshold Too Low**: Increase thresholds for problematic methods
2. **Uncommon Valid Values**: Update validation rules or patterns
3. **Data Distribution Changes**: Retrain ML models
4. **Ambiguous Cases**: Document and create exceptions

### Investigation Process

```python
# Load false positives
import pandas as pd
import json

predictions = pd.read_csv('predictions.csv')
false_positives = predictions[
    (predictions['detection_result'] == True) & 
    (predictions['ground_truth'] == False)
]

# Analyze patterns
print("False positives by field:")
print(false_positives['field'].value_counts())

print("\nFalse positives by method:")
print(false_positives['detection_method'].value_counts())

# Export for manual review
false_positives.to_csv('false_positives_review.csv', index=False)
```

## Improving Performance

### Based on Evaluation Results

1. **Low Precision**: 
   - Increase detection thresholds
   - Refine validation rules
   - Add exceptions for valid edge cases

2. **Low Recall**:
   - Lower detection thresholds
   - Add new validation rules
   - Enable additional detection methods

3. **Imbalanced Performance**:
   - Use weighted combination
   - Adjust method-specific thresholds
   - Focus on high-performing methods

### Continuous Improvement Process

1. Regular evaluation (weekly/monthly)
2. Review false positives and negatives
3. Update rules and thresholds
4. Retrain ML models if needed
5. Document changes and impacts

## Next Steps

- [Optimizing Detection Weights](optimizing-weights.md)
- [Advanced Configuration](../configuration/advanced-settings.md)
- [Custom Evaluation Metrics](../development/custom-metrics.md)