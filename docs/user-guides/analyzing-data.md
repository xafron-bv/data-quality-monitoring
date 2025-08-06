# Analyzing Your Data

This guide walks you through analyzing your data to understand its structure and identify the best detection strategies.

## Prerequisites

- Data Quality Detection System installed
- Access to your data file (CSV format)
- Basic understanding of your data fields

## Step 1: Explore Your Data Structure

Before running any detection, it's important to understand your data:

```bash
# View the first few rows of your data
head -n 10 your_data.csv

# Count the number of records
wc -l your_data.csv
```

## Step 2: Analyze Individual Columns

Use the `analyze-column` command to understand the characteristics of specific fields:

```bash
python main.py analyze-column your_data.csv column_name
```

This command will show you:
- Unique value distribution
- Common patterns
- Potential anomalies
- Recommended detection methods

### Example: Analyzing Product Names

```bash
python main.py analyze-column products.csv product_name
```

Output includes:
- Total unique values
- Top 10 most common values
- Pattern analysis results
- Suggested thresholds

## Step 3: Identify Key Fields

Not all fields are equally important for quality detection. Focus on:

1. **Business-critical fields**: Fields that directly impact operations
2. **High-variability fields**: Fields with many unique values
3. **Structured fields**: Fields with expected patterns (SKUs, codes, etc.)

## Step 4: Review Field Mappings

Check your brand configuration to ensure fields are properly mapped:

```bash
# View current field mappings
cat brand_configs/your_brand.json | jq '.field_mappings'
```

## Step 5: Run Initial Detection

Start with your data to test your configuration:

```bash
python main.py single-demo \
    --data-file your_data.csv \
    --enable-validation \
    --enable-pattern
```

Note: To test on a smaller sample, consider creating a subset of your data file first using standard tools like `head -n 100 your_data.csv > sample_data.csv`.

## Next Steps

- [Configuring Detection Methods](../detection-methods/overview.md)
- [Setting Up Brand Configuration](../configuration/brand-config.md)
- [Running Full Detection](running-detection.md)

## Tips

- Start with rule-based validation for high-confidence detection
- Use pattern-based detection for structured fields
- Reserve ML/LLM detection for complex semantic analysis
- Always validate results on a sample before full-scale processing