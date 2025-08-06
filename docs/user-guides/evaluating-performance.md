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
python main.py multi-eval your_data.csv \
    --field material \
    --num-samples 100 \
    --output-dir evaluation_results
```

### Field-Specific Evaluation

Evaluate specific fields with different detectors:

```bash
python main.py multi-eval your_data.csv \
    --field color_name \
    --ml-detector \
    --run all \
    --num-samples 50
```

### Synthetic Error Evaluation

The multi-eval command automatically injects errors for evaluation:

```bash
python main.py multi-eval clean_data.csv \
    --field material \
    --error-probability 0.2 \
    --max-errors 3 \
    --num-samples 100
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
python main.py multi-eval daily_data.csv \
    --field material \
    --num-samples 50 \
    --output-dir monitoring/$DATE

# Analyze results over time using standard tools
# or custom scripts to parse the JSON output files
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
python main.py multi-eval benchmark_data.csv \
    --field material \
    --output-dir benchmark_results \
    --num-samples 100
```

Note: The current implementation doesn't support ground truth labels. Evaluation is done by injecting synthetic errors and measuring detection performance.

### Compare Configurations

Test different threshold settings:

```bash
# Conservative configuration
python main.py multi-eval benchmark_data.csv \
    --field material \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.8 \
    --output-dir results/conservative

# Aggressive configuration  
python main.py multi-eval benchmark_data.csv \
    --field material \
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
- [Brand Configuration](../configuration/brand-config.md)
- [Adding New Fields](../development/new-fields.md)