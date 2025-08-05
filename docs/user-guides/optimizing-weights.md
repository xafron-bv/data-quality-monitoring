# Optimizing Detection Weights

This guide explains how to optimize the weighted combination of detection methods based on their historical performance.

## Understanding Weighted Combination

The system can combine multiple detection methods using optimized weights that reflect each method's effectiveness for specific fields. This improves overall detection accuracy by relying more heavily on methods that perform well for particular data types.

## Generating Detection Weights

### From Evaluation Results

After running an evaluation, generate optimized weights:

```bash
python single_sample_multi_field_demo/generate_detection_weights.py \
    -i evaluation_results/report.json \
    -o detection_weights.json
```

### Understanding the Output

The generated weights file looks like:

```json
{
  "field_weights": {
    "material": {
      "validation": 0.45,
      "pattern_based": 0.35,
      "ml_based": 0.15,
      "llm_based": 0.05
    },
    "color_name": {
      "validation": 0.30,
      "pattern_based": 0.25,
      "ml_based": 0.35,
      "llm_based": 0.10
    }
  },
  "default_weights": {
    "validation": 0.40,
    "pattern_based": 0.30,
    "ml_based": 0.20,
    "llm_based": 0.10
  }
}
```

### Weight Interpretation

- Higher weights indicate better historical performance
- Weights sum to 1.0 for each field
- Default weights apply to fields without specific optimization

## Using Optimized Weights

### Apply Weights in Detection

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --use-weighted-combination \
    --weights-file detection_weights.json \
    --enable-validation \
    --enable-pattern \
    --enable-ml
```

### Comparing With and Without Weights

Run a comparison to see the improvement:

```bash
# Without weights
python main.py single-demo \
    --data-file test_data.csv \
    --output-dir results/no_weights \
    --enable-validation \
    --enable-pattern

# With weights
python main.py single-demo \
    --data-file test_data.csv \
    --output-dir results/with_weights \
    --use-weighted-combination \
    --weights-file detection_weights.json \
    --enable-validation \
    --enable-pattern
```

## Advanced Weight Generation

### Custom Weight Calculation

Create custom weights based on specific metrics:

```python
import json

# Load evaluation results
with open('evaluation_results/detailed_report.json', 'r') as f:
    results = json.load(f)

# Custom weight calculation
weights = {"field_weights": {}}

for field, metrics in results['by_field'].items():
    field_weights = {}
    
    for method, perf in metrics['by_method'].items():
        # Weight by F1 score and confidence
        f1_score = perf.get('f1_score', 0)
        avg_confidence = perf.get('avg_confidence', 0)
        
        # Custom formula: prioritize F1 but consider confidence
        weight = (0.7 * f1_score) + (0.3 * avg_confidence)
        field_weights[method] = weight
    
    # Normalize weights
    total = sum(field_weights.values())
    if total > 0:
        field_weights = {k: v/total for k, v in field_weights.items()}
    
    weights['field_weights'][field] = field_weights

# Save custom weights
with open('custom_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)
```

### Weight Optimization Strategies

#### 1. Precision-Focused Weights

For applications where false positives are costly:

```python
# Prioritize high-precision methods
for field, metrics in results['by_field'].items():
    field_weights = {}
    for method, perf in metrics['by_method'].items():
        # Weight heavily by precision
        weight = perf.get('precision', 0) ** 2
        field_weights[method] = weight
```

#### 2. Recall-Focused Weights

For applications where missing errors is costly:

```python
# Prioritize high-recall methods
for field, metrics in results['by_field'].items():
    field_weights = {}
    for method, perf in metrics['by_method'].items():
        # Weight heavily by recall
        weight = perf.get('recall', 0) ** 2
        field_weights[method] = weight
```

#### 3. Balanced Weights

For general-purpose detection:

```python
# Use F1 score for balanced performance
for field, metrics in results['by_field'].items():
    field_weights = {}
    for method, perf in metrics['by_method'].items():
        weight = perf.get('f1_score', 0)
        field_weights[method] = weight
```

## Iterative Weight Refinement

### Continuous Improvement Process

1. **Initial Weights**: Start with default or generated weights
2. **Evaluation**: Run detection with current weights
3. **Analysis**: Review performance metrics
4. **Adjustment**: Refine weights based on results
5. **Validation**: Test on holdout data

### Tracking Weight Performance

Create a weight performance log:

```json
{
  "weight_history": [
    {
      "date": "2024-01-15",
      "weights_file": "weights_v1.json",
      "performance": {
        "overall_f1": 0.85,
        "precision": 0.88,
        "recall": 0.82
      }
    },
    {
      "date": "2024-01-22",
      "weights_file": "weights_v2.json",
      "performance": {
        "overall_f1": 0.87,
        "precision": 0.89,
        "recall": 0.85
      }
    }
  ]
}
```

## Best Practices

### 1. Regular Updates

- Re-evaluate weights monthly or when data patterns change
- Use recent evaluation data (last 30-90 days)
- Version control weight files

### 2. Field-Specific Optimization

- Some fields benefit from rule-based detection
- Others need semantic understanding (ML/LLM)
- Let the data guide weight distribution

### 3. Method Selection

- Don't force all methods on all fields
- Set very low weights (< 0.05) to effectively disable methods
- Consider computational cost vs. benefit

### 4. Validation

- Always test new weights on holdout data
- Compare against baseline (equal weights)
- Monitor for overfitting to evaluation data

## Troubleshooting

### Weights Not Improving Performance

1. Check if evaluation data is representative
2. Ensure sufficient data for each field/method
3. Verify methods are properly configured
4. Consider method limitations for specific fields

### Unstable Weights

1. Increase evaluation sample size
2. Use rolling average of multiple evaluations
3. Set minimum weight thresholds
4. Apply smoothing to weight changes

## Next Steps

- [Advanced Configuration](../configuration/advanced-settings.md)
- [Custom Detection Methods](../development/custom-detectors.md)
- [Performance Monitoring](../operations/monitoring.md)