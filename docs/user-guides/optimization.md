# Performance Optimization Guide

This guide covers evaluating detection performance and optimizing the system for your specific data.

## Performance Evaluation

### Understanding Metrics

The system tracks key performance metrics:

- **Precision**: How many detections were correct (true positives / all positives)
- **Recall**: How many errors were caught (true positives / all actual errors)  
- **F1 Score**: Harmonic mean of precision and recall
- **Detection Rate**: Percentage of records flagged
- **Confidence Distribution**: Spread of confidence scores

### Running Evaluation

#### Basic Evaluation

Use multi-eval for systematic performance testing:

```bash
python main.py multi-eval your_data.csv \
    --field material \
    --num-samples 100 \
    --output-dir evaluation_results
```

#### Field-Specific Evaluation

Test specific fields with different detectors:

```bash
python main.py multi-eval your_data.csv \
    --field color_name \
    --ml-detector \
    --run all \
    --num-samples 50
```

#### Synthetic Error Testing

Multi-eval automatically injects errors for evaluation:

```bash
python main.py multi-eval clean_data.csv \
    --field material \
    --error-probability 0.2 \
    --max-errors 3 \
    --num-samples 100
```

### Analyzing Evaluation Results

Review the generated reports to understand:
- Detection accuracy per field
- Method effectiveness comparison
- Error type distribution
- Confidence score calibration

## Weighted Combination Optimization

### Understanding Weighted Combination

The system can combine detection methods using optimized weights based on their effectiveness for specific fields. This improves accuracy by relying more on methods that perform well for particular data types.

### Generating Detection Weights

After evaluation, generate optimized weights:

```bash
python single_sample_multi_field_demo/generate_detection_weights.py \
    -i evaluation_results/report.json \
    -o detection_weights.json
```

### Weight File Structure

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

### Using Optimized Weights

Apply weights in detection:

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --use-weighted-combination \
    --weights-file detection_weights.json
```

## Threshold Optimization

### Finding Optimal Thresholds

Use ML curves to find the best thresholds:

```bash
python main.py ml-curves your_data.csv \
    --fields material color_name \
    --output-dir threshold_analysis
```

This generates:
- Precision-recall curves
- F1 score vs threshold plots
- Recommended threshold values

### Applying Optimized Thresholds

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.75 \
    --ml-threshold 0.82 \
    --llm-threshold 0.65
```

## Performance Tuning Strategies

### 1. Speed Optimization

For faster processing:
- Use `--core-fields-only` for essential fields
- Disable expensive methods (LLM) when not needed
- Process in batches for large datasets
- Enable GPU acceleration for ML/LLM

### 2. Accuracy Optimization

For better detection:
- Generate and use weighted combinations
- Fine-tune thresholds per field
- Train custom ML models
- Update validation rules regularly

### 3. Memory Optimization

For large datasets:
- Process fields sequentially
- Reduce batch sizes
- Use validation-only for initial screening
- Split data into smaller chunks

## Continuous Improvement Workflow

### 1. Baseline Establishment

```bash
# Initial evaluation
python main.py multi-eval baseline_data.csv \
    --field all \
    --num-samples 200 \
    --output-dir baseline_results
```

### 2. Weight Generation

```bash
# Generate optimized weights
python single_sample_multi_field_demo/generate_detection_weights.py \
    -i baseline_results/report.json \
    -o weights_v1.json
```

### 3. Performance Monitoring

```bash
# Regular evaluation with weights
python main.py single-demo \
    --data-file daily_data.csv \
    --use-weighted-combination \
    --weights-file weights_v1.json \
    --output-dir monitoring/$(date +%Y%m%d)
```

### 4. Periodic Re-optimization

Re-evaluate and update weights quarterly or when:
- Data patterns change significantly
- New fields are added
- Detection accuracy drops
- Business requirements change

## Best Practices

### For Evaluation
- Use representative data samples
- Test with realistic error rates
- Evaluate all critical fields
- Document baseline performance

### For Optimization
- Start with default weights
- Optimize incrementally
- Validate improvements
- Keep historical weights for comparison

### For Production
- Monitor detection rates
- Track false positive trends
- Update weights periodically
- Maintain separate weights for different data types

## Troubleshooting Performance Issues

### Low Precision (Too Many False Positives)
- Increase detection thresholds
- Review and update validation rules
- Reduce weights for noisy methods
- Consider field-specific thresholds

### Low Recall (Missing Real Errors)
- Lower detection thresholds
- Enable additional detection methods
- Increase weights for effective methods
- Add more validation rules

### Slow Processing
- Profile to identify bottlenecks
- Disable unnecessary methods
- Optimize batch sizes
- Consider parallel processing

### Unstable Results
- Check for data quality issues
- Verify consistent preprocessing
- Use larger evaluation samples
- Average results across multiple runs