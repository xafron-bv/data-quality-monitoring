# Column Analysis Tool - How-To Guide

## Overview

The Column Analysis Tool (`analyze_column.py`) is a utility script for analyzing the unique values and distribution of data in specific columns of a CSV file. It's particularly useful for understanding your data before creating validation rules or training anomaly detection models.

## Features

- **Value Distribution Analysis**: See all unique values and their frequencies
- **Basic Statistics**: Get count, unique count, and basic stats for numeric fields
- **Missing Value Detection**: Identify empty or null values
- **Pattern Recognition**: Spot common patterns in your data
- **Brand-Aware Mapping**: Use brand configurations to map field names

## Prerequisites

1. Python 3.8+
2. Required packages (install from root directory):
   ```bash
   pip install -r requirements.txt
   ```
3. CSV file to analyze
4. Brand configuration (optional, default: "fashion_brand")

## Basic Usage

### 1. Navigate to the column_analysis directory:
```bash
cd column_analysis
```

### 2. Analyze a specific column:
```bash
python analyze_column.py ../data/products.csv size
```

### 3. View the output:
The analysis is printed to the console, showing:
- Total number of rows
- Number of unique values
- Value frequency distribution
- Missing value count

## Command Line Options

```bash
python analyze_column.py <csv_file> <field_name> [options]
```

### Required Arguments:
- `csv_file`: Path to the CSV file to analyze
- `field_name`: The field/column name to analyze (logical name, not CSV column name)

### Optional Arguments:
- `--brand`: Brand configuration to use (default: "fashion_brand")
- `--limit`: Limit the number of unique values to display (default: show all)
- `--sort`: Sort order for values ("frequency" or "value", default: frequency)
- `--output`: Save analysis to a file instead of printing

## Examples

### Example 1: Basic column analysis
```bash
python analyze_column.py ../data/products.csv color_name
```

### Example 2: Analyze with different brand configuration
```bash
python analyze_column.py ../data/inventory.csv size --brand custom_brand
```

### Example 3: Limit output to top 20 values
```bash
python analyze_column.py ../data/products.csv category --limit 20
```

### Example 4: Sort by value instead of frequency
```bash
python analyze_column.py ../data/products.csv material --sort value
```

### Example 5: Save analysis to file
```bash
python analyze_column.py ../data/products.csv season --output season_analysis.txt
```

## Understanding the Output

### Sample Output:
```
Analyzing field: size
CSV column: Size

Total rows: 10000
Non-null values: 9950
Missing values: 50 (0.5%)

Unique values: 15

Value distribution:
M          : 2500 (25.0%)
L          : 2000 (20.0%)
S          : 1800 (18.0%)
XL         : 1500 (15.0%)
XS         : 1000 (10.0%)
XXL        : 500 (5.0%)
2XL        : 300 (3.0%)
3XL        : 200 (2.0%)
[empty]    : 50 (0.5%)
XXXL       : 50 (0.5%)
...
```

### Key Metrics:
- **Total rows**: Total number of records in the CSV
- **Non-null values**: Records with actual data
- **Missing values**: Empty or null entries
- **Unique values**: Number of distinct values
- **Value distribution**: Frequency of each value

## Using Analysis for Validation Rules

### 1. Identify Valid Values:
```bash
# Analyze to find all valid sizes
python analyze_column.py data.csv size --output valid_sizes.txt

# Use this to create validation rules
```

### 2. Spot Data Quality Issues:
- Typos: "Smal" instead of "Small"
- Inconsistencies: "XL" vs "X-Large"
- Invalid entries: "N/A", "Unknown"

### 3. Define Rules Based on Patterns:
```python
# If analysis shows sizes follow pattern
valid_patterns = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
```

## Using Analysis for Anomaly Detection

### 1. Identify Rare Values:
Values with very low frequency might be anomalies:
```
Ultra-Rare-Size: 1 (0.01%)  # Potential anomaly
```

### 2. Establish Baselines:
Normal distribution helps set anomaly thresholds:
```
Normal range: S, M, L, XL (90% of data)
Outliers: Everything else
```

### 3. Pattern Detection:
Look for values that don't match expected patterns:
```
Expected: Numeric sizes (36, 38, 40)
Anomaly: Mixed formats (36, Medium, L)
```

## Batch Analysis

Analyze multiple fields at once:
```bash
#!/bin/bash
FIELDS=("size" "color_name" "material" "category" "season")

for field in "${FIELDS[@]}"; do
    echo "Analyzing $field..."
    python analyze_column.py data.csv $field --output "analysis/${field}_analysis.txt"
done
```

## Advanced Usage

### Custom Analysis Script:
```python
import sys
sys.path.append('..')
from analyze_column import analyze_field_values
from field_mapper import FieldMapper

# Custom analysis
mapper = FieldMapper.from_brand("fashion_brand")
results = analyze_field_values("data.csv", "size", mapper)

# Process results programmatically
for value, count in results.items():
    if count < 10:
        print(f"Rare value detected: {value}")
```

### Integration with Other Tools:
```bash
# Analyze first, then create rules
python analyze_column.py data.csv material --output material_analysis.txt

# Use analysis to inform validator creation
cd ../validators/material
# Edit validation rules based on analysis
```

## Troubleshooting

### Field Not Found:
- Check the field name matches brand configuration
- Use logical field name, not CSV column name
- Verify brand configuration is correct

### Empty Results:
- Check if CSV file has data
- Verify column actually exists
- Look for case sensitivity issues

### Memory Issues:
- Use `--limit` to reduce output size
- Process large files in chunks
- Consider sampling the data first

## Best Practices

1. **Analyze Before Rules**: Always analyze data before creating validation rules
2. **Regular Checks**: Re-analyze periodically to catch data drift
3. **Document Findings**: Save analysis results for reference
4. **Compare Periods**: Analyze same field across different time periods
5. **Cross-Reference**: Compare related fields (e.g., category vs. subcategory)

## Tips for Effective Analysis

### 1. Start Broad:
First analyze without limits to see full distribution

### 2. Focus on Anomalies:
Look for values that seem out of place

### 3. Check Relationships:
Analyze related columns together

### 4. Version Control:
Save analysis outputs with timestamps

### 5. Automate Regular Checks:
Set up scheduled analysis runs

## Next Steps

After analyzing columns:

1. **Create Validation Rules**:
   - Use found values to define valid sets
   - Create patterns for format validation

2. **Train Anomaly Detectors**:
   - Use frequency data for threshold setting
   - Identify normal vs. anomalous patterns

3. **Generate Test Data**:
   - Use distribution for realistic test data
   - Include edge cases found in analysis

4. **Monitor Changes**:
   - Set up alerts for new values
   - Track distribution changes over time

## Support

For issues or questions:
1. Check the error messages for specific issues
2. Verify CSV file format and encoding
3. Ensure brand configuration is properly set up
4. Consult the main project documentation