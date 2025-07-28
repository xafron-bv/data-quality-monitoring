# Detection Weights Generator - How-To Guide

## Overview

The Detection Weights Generator (`generate_detection_weights.py`) analyzes performance results from data quality detection runs and generates optimized weights for each detection method (validation, pattern-based, ML-based) on a per-field basis. These weights enable the weighted combination detection approach, which dynamically prioritizes the most effective detection method for each field.

## Purpose

Instead of using a fixed priority (validation > pattern > ML), the weighted combination approach:
- Assigns weights based on actual performance metrics
- Adapts to field-specific characteristics
- Improves overall detection accuracy
- Reduces false positives while maintaining high recall

## Prerequisites

1. Python 3.8+
2. A unified report JSON file from:
   - Demo analysis runs (`demo_analysis_unified_report.json`)
   - Multi-sample evaluations
   - Any tool that generates compatible performance metrics

## Basic Usage

### 1. Navigate to the weights_generation directory:
```bash
cd weights_generation
```

### 2. Generate weights from a demo report:
```bash
python generate_detection_weights.py -i ../demo_analysis/demo_results/demo_analysis_unified_report.json
```

### 3. View generated weights:
```bash
cat detection_weights.json
```

### 4. Use the viewer:
- Open `detection_weights_viewer.html` in a browser
- Upload the generated `detection_weights.json` file

## Command Line Options

```bash
python generate_detection_weights.py [options]
```

### Required Arguments:
- `-i, --input-file`: Path to unified report JSON file with performance data

### Optional Arguments:
- `-o, --output-file`: Output file for weights (default: `./detection_weights.json`)
- `-b, --baseline-weight`: Weight for untrained methods (default: 0.1)
- `-v, --verbose`: Enable verbose output with detailed insights

## How It Works

### Weight Calculation Algorithm:

1. **Extract Performance Metrics**:
   - Precision, recall, and F1 scores for each detection method
   - Per-field performance data

2. **Calculate Weights**:
   ```
   weight = (F1_score * 0.6) + (precision * 0.2) + (recall * 0.2)
   ```
   - F1 score is weighted highest (60%) as it balances precision and recall
   - Precision and recall each contribute 20%

3. **Normalize Weights**:
   - Weights are normalized so they sum to 1.0 for each field
   - Methods without data get the baseline weight

4. **Generate Insights**:
   - Identifies dominant methods per field
   - Provides recommendations for optimization

## Examples

### Example 1: Basic weight generation
```bash
python generate_detection_weights.py \
    -i ../demo_analysis/demo_results/demo_analysis_unified_report.json
```

### Example 2: Custom output with verbose insights
```bash
python generate_detection_weights.py \
    -i evaluation_results/unified_report.json \
    -o custom_weights.json \
    -v
```

### Example 3: Lower baseline weight for untrained methods
```bash
python generate_detection_weights.py \
    -i report.json \
    -b 0.05 \
    -v
```

### Example 4: Generate weights from multiple evaluations
```bash
# First, combine multiple reports (custom script needed)
python combine_reports.py report1.json report2.json -o combined_report.json

# Then generate weights
python generate_detection_weights.py -i combined_report.json
```

## Output Format

The generated `detection_weights.json` contains:

```json
{
  "field_weights": {
    "size": {
      "validation": 0.7,
      "pattern_based": 0.2,
      "ml_based": 0.1
    },
    "color_name": {
      "validation": 0.3,
      "pattern_based": 0.2,
      "ml_based": 0.5
    }
  },
  "weight_summary": {
    "size": {
      "dominant_method": "validation",
      "rationale": "Validation shows highest F1 score (0.95)"
    }
  },
  "performance_insights": {
    "size": {
      "observation": "Validation performs exceptionally well",
      "recommendation": "Rely primarily on validation rules"
    }
  },
  "metadata": {
    "source_file": "report.json",
    "generated_at": "2024-01-15T10:30:00Z",
    "baseline_weight": 0.1
  }
}
```

## Interpreting Results

### Weight Distribution:
- **High weight (> 0.6)**: Method is highly effective for this field
- **Medium weight (0.3-0.6)**: Method provides moderate value
- **Low weight (< 0.3)**: Method is less effective or untrained
- **Baseline weight**: Method has no performance data

### Common Patterns:

1. **Validation Dominant**:
   - Structured fields (size, SKU, dates)
   - Fields with clear rules
   - High precision requirements

2. **ML Dominant**:
   - Free-text fields (descriptions, titles)
   - Semantic consistency checks
   - Context-dependent validation

3. **Balanced Weights**:
   - Complex fields needing multiple approaches
   - Fields with diverse error types

## Using Generated Weights

### In Demo Analysis:
```bash
cd ../demo_analysis
python single_sample_multi_field_demo.py data.csv \
    --use-weighted-combination \
    --weights-file ../weights_generation/detection_weights.json
```

### In Custom Scripts:
```python
import json

# Load weights
with open('detection_weights.json', 'r') as f:
    weights_data = json.load(f)

# Get weights for a field
field_weights = weights_data['field_weights']['size']
validation_weight = field_weights['validation']
```

## Optimizing Weights

### 1. Collect More Data:
- Run evaluations on larger datasets
- Test with real-world error patterns
- Include edge cases in testing

### 2. Adjust Baseline Weight:
- Lower (0.05) for strict mode
- Higher (0.2) for exploratory mode
- Zero to disable untrained methods

### 3. Combine Multiple Reports:
- Average weights from multiple runs
- Weight recent reports higher
- Consider seasonal variations

### 4. Field-Specific Tuning:
- Override generated weights for critical fields
- Set minimum thresholds
- Add business logic constraints

## Troubleshooting

### No Performance Data:
- Ensure input report contains metrics
- Check field names match exactly
- Verify detection methods were enabled

### Unexpected Weights:
- Review input report for anomalies
- Check if sufficient test samples were used
- Verify error injection was balanced

### Low Overall Performance:
- Consider training better models
- Review and update validation rules
- Add more error patterns to testing

## Best Practices

1. **Regular Updates**: Regenerate weights periodically as models improve
2. **A/B Testing**: Compare weighted vs. fixed priority performance
3. **Monitor Drift**: Track weight changes over time
4. **Document Changes**: Keep changelog of weight updates
5. **Validate Impact**: Measure real-world performance improvements

## Advanced Usage

### Custom Weight Formulas:
Modify the `generate_weights_from_performance` function to use different formulas:
```python
# Example: Prioritize recall for critical fields
if field_name in critical_fields:
    weight = (f1 * 0.4) + (precision * 0.1) + (recall * 0.5)
```

### Weight Constraints:
Add business rules to the weight generation:
```python
# Ensure validation always has minimum weight for compliance
if field_name in compliance_fields:
    weights['validation'] = max(weights['validation'], 0.5)
```

### Integration with CI/CD:
```bash
#!/bin/bash
# Auto-generate weights after model training
python train_models.py
python run_evaluation.py
python generate_detection_weights.py -i evaluation_results/report.json
git add detection_weights.json
git commit -m "Update detection weights after model training"
```

## Support

For issues or questions:
1. Check that input reports have the expected format
2. Review verbose output for detailed diagnostics
3. Consult the performance insights in the output
4. See main project documentation for report format specifications