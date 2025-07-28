# Multi-Sample Evaluation Tool - How-To Guide

## Overview

The Multi-Sample Evaluation Tool (`multi_sample_evaluation.py`) is designed to systematically evaluate the performance of validation rules, anomaly detectors, and ML-based detection methods. It generates multiple synthetic samples with known errors and anomalies, then measures how well each detection method identifies these issues.

## Features

- **Multi-Sample Generation**: Creates multiple test samples with controlled error injection
- **Comprehensive Evaluation**: Tests validation, anomaly detection, and ML-based methods
- **Performance Metrics**: Calculates precision, recall, F1 scores, and confusion matrices
- **Comparative Analysis**: Compares performance across different detection methods
- **Detailed Reporting**: Generates comprehensive evaluation reports

## Prerequisites

1. Python 3.8+
2. Required packages (install from root directory):
   ```bash
   pip install -r requirements.txt
   ```
3. Input CSV file with clean product data
4. Configured validators and anomaly detectors for your fields
5. Trained ML models (if using ML detection)

## Basic Usage

### 1. Navigate to the ml_evaluation directory:
```bash
cd ml_evaluation
```

### 2. Run evaluation for a specific field:
```bash
python multi_sample_evaluation.py ../data/clean_products.csv material
```

### 3. View results:
- Reports are saved in `ml_evaluation/evaluation_results/`
- Open `unified_report_viewer.html` in a browser
- Upload the generated summary report JSON file

## Command Line Options

```bash
python multi_sample_evaluation.py <csv_file> <field_name> [options]
```

### Required Arguments:
- `csv_file`: Path to the input CSV file (should contain clean data)
- `field_name`: The field/column to evaluate

### Optional Arguments:

#### Evaluation Options:
- `--run`: Which analysis to run (default: "both")
  - `validation`: Only validation rules
  - `anomaly`: Only anomaly detection
  - `ml`: Only ML detection
  - `both`: Validation + anomaly detection
  - `all`: All three methods
- `--num-samples`: Number of samples to generate (default: 32)
- `--max-errors`: Maximum errors per sample (default: 3)
- `--output-dir`: Output directory (default: "./evaluation_results")

#### Error Control:
- `--ignore-errors`: List of error rules to ignore
- `--ignore-fp`: Ignore false positives in evaluation
- `--brand`: Brand configuration (default: "fashion_brand")

#### Advanced Options:
- `--debug`: Enable debug logging
- `--llm-api-key`: API key for LLM detection
- `--enable-llm`: Enable LLM-based detection
- `--llm-threshold`: LLM confidence threshold (default: 0.9)

## Examples

### Example 1: Basic validation evaluation
```bash
python multi_sample_evaluation.py ../data/products.csv size --run validation
```

### Example 2: Comprehensive evaluation with all methods
```bash
python multi_sample_evaluation.py ../data/products.csv category \
    --run all \
    --num-samples 50 \
    --max-errors 5
```

### Example 3: ML evaluation with custom output
```bash
python multi_sample_evaluation.py ../data/products.csv color_name \
    --run ml \
    --output-dir ml_evaluation_color \
    --debug
```

### Example 4: Evaluation ignoring specific errors
```bash
python multi_sample_evaluation.py ../data/products.csv material \
    --ignore-errors inject_typo inject_unicode_error \
    --ignore-fp
```

### Example 5: LLM-enabled evaluation
```bash
python multi_sample_evaluation.py ../data/products.csv season \
    --run all \
    --enable-llm \
    --llm-api-key your_api_key \
    --llm-threshold 0.85
```

## Output Structure

The tool creates the following directory structure:

```
evaluation_results/
├── samples/
│   ├── sample_0/
│   │   ├── corrupted_data.csv
│   │   └── error_log.json
│   ├── sample_1/
│   │   └── ...
│   └── ...
├── evaluation_summary.json
├── evaluation_summary.txt
└── [field]_evaluation_unified_report.json
```

### Key Output Files:

1. **evaluation_summary.txt**: Human-readable summary of results
2. **evaluation_summary.json**: Detailed metrics in JSON format
3. **[field]_evaluation_unified_report.json**: Report for the unified viewer
4. **samples/**: Individual test samples with error logs

## Understanding the Results

### Performance Metrics:

- **True Positives (TP)**: Correctly detected errors
- **False Positives (FP)**: Incorrectly flagged clean data
- **False Negatives (FN)**: Missed errors
- **Precision**: TP / (TP + FP) - accuracy of positive predictions
- **Recall**: TP / (TP + FN) - completeness of detection
- **F1 Score**: Harmonic mean of precision and recall

### Evaluation Summary Sections:

1. **Overall Statistics**: Total samples, errors injected, detection rates
2. **Detection Performance**: Metrics for each detection method
3. **Error Type Analysis**: Performance breakdown by error type
4. **False Positive Analysis**: Common false positive patterns
5. **False Negative Analysis**: Commonly missed errors

## Interpreting Results

### Good Performance Indicators:
- High F1 score (> 0.8)
- Balanced precision and recall
- Low false positive rate
- Consistent performance across error types

### Warning Signs:
- Very low recall (< 0.5) - missing many errors
- Very low precision (< 0.5) - too many false alarms
- High variance across error types
- Degraded performance with multiple errors

## Batch Evaluation

To evaluate multiple fields, use a shell script:

```bash
#!/bin/bash
FIELDS=("size" "color_name" "material" "category" "season")
CSV_FILE="../data/products.csv"

for field in "${FIELDS[@]}"; do
    echo "Evaluating $field..."
    python multi_sample_evaluation.py "$CSV_FILE" "$field" \
        --run all \
        --output-dir "batch_results/$field"
done
```

## Troubleshooting

### Common Issues:

1. **No Validators Found**:
   - Check that validators exist in `../validators/[field]/`
   - Verify the field name matches the validator directory

2. **ML Model Not Found**:
   - Ensure ML models are trained for the field
   - Check model paths in configuration

3. **Low Performance**:
   - Review error injection rules
   - Check if validators are too strict/lenient
   - Verify training data quality for ML models

4. **Memory Issues**:
   - Reduce `--num-samples`
   - Process fields individually
   - Use `--run` to test one method at a time

### Debug Mode:
Enable detailed logging:
```bash
python multi_sample_evaluation.py ../data/products.csv material --debug
```

## Advanced Usage

### Custom Error Rules:
1. Add rules to `../validators/[field]/error_rules.json`
2. Test with small sample size first
3. Adjust rule parameters based on results

### Custom Anomaly Rules:
1. Add rules to `../anomaly_detectors/pattern_based/[field]/anomaly_rules.json`
2. Ensure rules don't overlap with validation rules
3. Test detection thresholds

### Integration with Other Tools:
- Use results to generate weights: `../weights_generation/generate_detection_weights.py`
- Compare with other methods: `../detection_comparison/detection_comparison.py`
- Run comprehensive demo: `../demo_analysis/single_sample_multi_field_demo.py`

## Best Practices

1. **Start Simple**: Test one field with one detection method
2. **Iterate**: Gradually increase complexity
3. **Baseline**: Establish performance baselines for each field
4. **Document**: Keep notes on what works for each field type
5. **Version Control**: Track changes to rules and configurations
6. **Regular Testing**: Re-evaluate when rules or models change

## Optimizing Performance

### For Better Recall:
- Add more error injection rules
- Lower detection thresholds
- Combine multiple detection methods

### For Better Precision:
- Refine validation rules
- Increase ML model confidence thresholds
- Add context-aware rules

### For Balanced Performance:
- Use weighted combination of methods
- Tune thresholds based on use case
- Consider field-specific approaches

## Support

For issues or questions:
1. Check error logs in the output directory
2. Review debug output with `--debug` flag
3. Consult field-specific validator documentation
4. Check the main project README for system requirements