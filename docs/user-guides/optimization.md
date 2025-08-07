# Performance Optimization Guide

This guide covers evaluating detection performance and optimizing the system for your specific data.

## Performance Evaluation

### Understanding Metrics

- Precision, Recall, F1 Score
- Detection Rate
- Confidence Distribution

### Running Evaluation

#### Basic Evaluation

```python
import subprocess
subprocess.run([
  "python", "main.py", "multi-eval",
  "your_data.csv", "--field", "material",
  "--num-samples", "100",
  "--output-dir", "evaluation_results",
])
```

#### Field-Specific Evaluation

```python
import subprocess
subprocess.run([
  "python", "main.py", "multi-eval",
  "your_data.csv", "--field", "color_name",
  "--ml-detector", "--run", "all",
  "--num-samples", "50",
])
```

## Weighted Combination Optimization

### Generating Detection Weights

After a single-demo run, use the unified report to generate optimized weights:

```python
import subprocess
subprocess.run([
  "python", "single_sample_multi_field_demo/generate_detection_weights.py",
  "-i", "results/demo_analysis_unified_report.json",
  "-o", "detection_weights.json",
])
```

### Weight Report Structure

```json
{
  "metadata": {"baseline_weight": 0.1, "generated_by": "generate_detection_weights.py"},
  "weights": {"material": {"pattern_based": 0.35, "ml_based": 0.5, "llm_based": 0.15}},
  "weight_summary": {"material": {"dominant_method": "ml_based", "dominant_weight": 0.5, "weights": {"pattern_based": 0.35, "ml_based": 0.5, "llm_based": 0.15}}},
  "performance_insights": {"material": ["ml_based: Good (F1=0.78)"]}
}
```

### Using Optimized Weights

```python
import subprocess
subprocess.run([
  "python", "main.py", "single-demo",
  "--data-file", "your_data.csv",
  "--use-weighted-combination",
  "--weights-file", "detection_weights.json",
])
```

## Threshold Optimization

Use ML curves to find the best thresholds:

```python
import subprocess
subprocess.run([
  "python", "main.py", "ml-curves", "your_data.csv",
  "--fields", "material color_name",
  "--output-dir", "threshold_analysis",
])
```

This generates precision-recall curves and recommends thresholds.

Apply optimized thresholds:

```python
import subprocess
subprocess.run([
  "python", "main.py", "single-demo",
  "--data-file", "your_data.csv",
  "--validation-threshold", "0.0",
  "--anomaly-threshold", "0.75",
  "--ml-threshold", "0.82",
  "--llm-threshold", "0.65",
])
```

## Tuning Strategies

- Speed: core fields only; disable LLM; batch processing; GPU for ML/LLM
- Accuracy: weighted combinations; field thresholds; custom ML models
- Memory: sequential fields; reduce batch size; split data

## Continuous Improvement

1. Baseline: multi-eval on representative data
2. Weight Generation: use `<sample>_unified_report.json`
3. Monitoring: regular single-demo with weights
4. Re-optimize when patterns or performance change

## Troubleshooting Performance

- Low Precision: raise thresholds; refine rules; reduce noisy method weights
- Low Recall: lower thresholds; enable additional methods; add rules
- Slow: disable unnecessary methods; optimize batch sizes; parallelize where safe
- Unstable: verify data quality; use larger evaluation samples