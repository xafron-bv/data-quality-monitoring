# Configuration Reference

This reference covers all configuration options for the Data Quality Detection System.

## Brand Configuration

Brand configurations define how the system maps your data columns to standard fields and sets brand-specific parameters.

### Location

Brand configurations are stored in the `brand_configs/` directory:

```
brand_configs/
├── esqualo.json      # Example brand config
└── your_brand.json   # Your custom brand config
```

### Configuration Structure

```json
{
    "brand_name": "your_brand",
    
    "field_mappings": {
        "material": "Material_Column",
        "color_name": "Color_Description",
        "category": "Product_Category",
        "size": "Size_Value",
        "product_name": "Product_Name",
        "product_id": "SKU",
        "description": "Long_Description"
    },
    
    "default_data_path": "data/your_data.csv",
    
    "custom_thresholds": {
        "validation_threshold": 0.0,
        "anomaly_threshold": 0.7,
        "ml_threshold": 0.7,
        "llm_threshold": 0.6
    }
}
```

### Field Mappings

Map your CSV column names to standard field types:

```json
"field_mappings": {
    "material": "Material_Column",    // Maps 'Material_Column' to standard 'material' field
    "color_name": "Color_Description" // Maps 'Color_Description' to standard 'color_name' field
}
```

#### Standard Fields

The system recognizes these standard field types:

- **Product Attributes**
  - `material` - Product material composition
  - `color_name` - Color descriptions
  - `size` - Size values
  - `category` - Product categories
  - `subcategory` - Product subcategories
  
- **Identifiers**
  - `product_name` - Product names
  - `product_id` - Product IDs/SKUs
  - `brand_name` - Brand names
  
- **Descriptions**
  - `description` - Product descriptions
  - `care_instructions` - Care/maintenance instructions
  
- **Business Fields**
  - `price` - Price values
  - `country` - Country codes/names
  - `gender` - Gender classifications

### Default Paths

```json
"default_data_path": "data/products.csv"  // Default input file for demos
```

### Custom Thresholds

Override global detection thresholds:

```json
"custom_thresholds": {
    "validation_threshold": 0.0,  // 0.0 = strictest
    "anomaly_threshold": 0.7,     // 0.0-1.0 range
    "ml_threshold": 0.7,          // 0.0-1.0 range
    "llm_threshold": 0.6          // 0.0-1.0 range
}
```

## Detection Configuration

### Validation Rules

Validation rules are defined in JSON files under `validators/<field_name>/error_messages.json`:

```json
{
    "EMPTY_VALUE": {
        "message": "Material value is empty",
        "severity": "high"
    },
    "INVALID_FORMAT": {
        "message": "Material format is invalid: {details}",
        "severity": "medium"
    }
}
```

### Pattern Rules

Pattern-based detection rules are in `anomaly_detectors/pattern_based/rules/<field_name>.json`:

```json
{
    "known_values": ["cotton", "polyester", "wool", "silk"],
    "patterns": [
        {
            "name": "percentage_pattern",
            "regex": "\\d+%\\s+\\w+",
            "description": "Percentage followed by material"
        }
    ],
    "suspicious_patterns": [
        {
            "pattern": "test|temp|xxx",
            "reason": "Likely test data"
        }
    ]
}
```

### ML Model Configuration

ML models are configured during training:

```bash
# Training configuration via CLI
python main.py ml-train data.csv \
    --fields "material color_name" \
    --use-hp-search \
    --hp-trials 20
```

Model artifacts are stored in:
- `anomaly_detectors/ml_based/models/<field_name>/`
- `anomaly_detectors/llm_based/models/<field_name>/`

## Environment Variables

Control system behavior with environment variables:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Memory limits (not currently implemented)
export MAX_WORKERS=4
export BATCH_SIZE=32
```

## Weight Configuration

Optimized detection weights are stored in JSON files:

```json
{
  "field_weights": {
    "material": {
      "validation": 0.45,
      "pattern_based": 0.35,
      "ml_based": 0.15,
      "llm_based": 0.05
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

Use with: `--use-weighted-combination --weights-file weights.json`

## Output Configuration

### Directory Structure

Output directories follow this structure:

```
output_dir/
├── report.json                  # Main detection report
├── viewer_report.json          # HTML viewer format
├── anomaly_summary.csv         # CSV summary
├── sample_with_errors.csv      # Data with injected errors
├── sample_with_results.csv     # Original data with results
└── confusion_matrix/           # Performance visualizations
    ├── overall_matrix.png
    └── per_field_matrix.png
```

### Report Configuration

Control report generation with CLI flags:
- `--generate-weights` - Generate weight recommendations
- `--core-fields-only` - Process only essential fields
- `--output-dir PATH` - Specify output location

## Performance Configuration

### Memory Management

- Use `--core-fields-only` to process only essential fields
- Adjust batch sizes for ML/LLM processing
- Process large files in chunks

### GPU Configuration

```bash
# Enable GPU
export CUDA_VISIBLE_DEVICES=0

# Disable GPU
python main.py single-demo --device cpu
```

### Parallel Processing

The system automatically parallelizes:
- Field processing within detection methods
- Multiple detection methods (when enabled)

## Creating a New Brand Configuration

1. **Copy the template**:
   ```bash
   cp brand_configs/esqualo.json brand_configs/new_brand.json
   ```

2. **Edit field mappings**:
   ```json
   {
     "brand_name": "new_brand",
     "field_mappings": {
       "material": "Your_Material_Column",
       "color_name": "Your_Color_Column"
     }
   }
   ```

3. **Test the configuration**:
   ```bash
   python main.py analyze-column your_data.csv Your_Material_Column
   ```

4. **Run detection**:
   ```bash
   python main.py single-demo --data-file your_data.csv
   ```

## Configuration Best Practices

### Field Mapping
- Map only columns that exist in your data
- Use exact column names (case-sensitive)
- Start with core fields, add more gradually

### Thresholds
- Start with default thresholds
- Adjust based on evaluation results
- Document threshold changes and reasons

### File Organization
- Keep one config file per brand/dataset
- Use descriptive file names
- Version control configuration files

### Testing
- Test configurations with small data samples first
- Verify field mappings with analyze-column
- Run evaluation to validate thresholds