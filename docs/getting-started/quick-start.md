# Quick Start Guide

Get up and running with the Data Quality Detection System in minutes. This guide will walk you through running your first data quality detection.

## Prerequisites

Before starting, ensure you have:
- ✅ Completed the [Installation Guide](installation.md)
- ✅ Python environment activated
- ✅ Sample data file (CSV format)

## Step 1: Prepare Your Data

The system expects data in CSV format with headers. Here's a sample structure:

```csv
product_id,product_name,material,color_name,category,size
P001,Cotton T-Shirt,100% Cotton,Navy Blue,Tops,M
P002,Leather Jacket,Genuine Leather,Black,Outerwear,L
P003,Silk Dress,100% Silk,Red,Dresses,S
```

Place your data file in the `data/` directory:

```bash
cp your_data.csv data/sample_data.csv
```

## Step 2: Run the Demo Script

The easiest way to start is with the single sample demo:

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file data/sample_data.csv \
    --output-dir results/quick_start
```

This will:
1. Load your data
2. Run all detection methods
3. Generate a comprehensive report
4. Save results to the output directory

## Step 3: Understanding the Output

After running, you'll find several files in the output directory:

```
results/quick_start/
├── sample_with_errors.csv      # Data with injected errors (for testing)
├── sample_with_results.csv     # Detection results per row
├── detection_report.json       # Detailed detection report
└── summary_report.txt          # Human-readable summary
```

### Reading the Summary Report

The summary report shows:
- Total records processed
- Errors detected by each method
- Performance metrics
- Field-level statistics

## Step 4: Visualize Results

Use the interactive HTML viewer:

1. Open `single_sample_multi_field_demo/data_quality_viewer.html` in a browser
2. Upload your CSV file: `sample_with_results.csv`
3. Upload the JSON report: `detection_report.json`
4. Explore the interactive visualizations

## Step 5: Customize Detection

### Enable Specific Methods

Run only validation and pattern-based detection:

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file data/sample_data.csv \
    --enable-validation \
    --enable-pattern \
    --disable-ml \
    --disable-llm
```

### Adjust Detection Thresholds

Make detection more or less sensitive:

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file data/sample_data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.8 \
    --ml-threshold 0.75
```

### Process Specific Fields

Focus on specific data fields:

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file data/sample_data.csv \
    --fields material color_name category
```

## Common Use Cases

### 1. Data Quality Check Before Import

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file import_data.csv \
    --enable-validation \
    --validation-threshold 0.0 \
    --output-dir results/import_check
```

### 2. Anomaly Detection in Production Data

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file production_data.csv \
    --enable-pattern \
    --enable-ml \
    --anomaly-threshold 0.7 \
    --output-dir results/anomaly_scan
```

### 3. Comprehensive Analysis with Error Injection

```bash
python single_sample_multi_field_demo/single_sample_multi_field_demo.py \
    --data-file clean_data.csv \
    --injection-intensity 0.2 \
    --enable-all \
    --output-dir results/full_analysis
```

## Understanding Detection Methods

### Validation (Rule-Based)
- **Best for**: Format errors, missing values, business rule violations
- **Confidence**: High (100%)
- **Example**: Invalid email format, empty required fields

### Pattern-Based Detection
- **Best for**: Unusual patterns, outliers from known values
- **Confidence**: Medium (70-80%)
- **Example**: Unusual color names, rare material combinations

### ML-Based Detection
- **Best for**: Semantic anomalies, contextual errors
- **Confidence**: Configurable (default 70%)
- **Example**: Mismatched product descriptions, category errors

### LLM-Based Detection
- **Best for**: Complex semantic understanding
- **Confidence**: Configurable (default 60%)
- **Example**: Logical inconsistencies, context-dependent errors

## Quick Debugging Tips

### No Errors Detected?
- Your data might be clean! Try with `--injection-intensity 0.3` to inject test errors
- Lower detection thresholds to be more sensitive
- Check if field mappings are correct in brand configuration

### Too Many False Positives?
- Increase detection thresholds
- Disable overly sensitive methods
- Review and update validation rules

### Performance Issues?
- Use `--core-fields-only` to process fewer fields
- Disable ML/LLM methods for faster processing
- Reduce sample size with `--sample-size 1000`

## Next Steps

Now that you've run your first detection:

1. **Learn More**: Read the [Basic Usage Guide](basic-usage.md)
2. **Configure**: Set up [Brand Configuration](../configuration/brand-config.md)
3. **Customize**: Add [New Fields](../development/new-fields.md)
4. **Evaluate**: Run [Performance Evaluation](../guides/evaluation.md)
5. **Deploy**: Follow the [Deployment Guide](../operations/deployment.md)

## Getting Help

- Check the full [CLI Reference](../reference/cli.md)
- Review [Configuration Options](../configuration/brand-config.md)
- See [Troubleshooting Guide](../operations/troubleshooting.md) for common issues