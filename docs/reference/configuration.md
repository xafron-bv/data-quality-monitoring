# Configuration Reference

This reference covers key configuration options for the Data Quality Detection System.

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
  "brand_name": "my_brand",
  "field_mappings": {
    "material": "Material",
    "color_name": "Color_Name",
    "category": "Category",
    "size": "Size",
    "description": "Long_Description"
  },
  "custom_thresholds": {
    "validation_threshold": 0.0,
    "anomaly_threshold": 0.8,
    "ml_threshold": 0.6
  },
  "enabled_fields": ["material", "color_name", "category", "size"],
  "field_variations": {
    "material": "baseline",
    "color_name": "baseline",
    "category": "baseline",
    "size": "baseline"
  }
}
```

### Field Mappings

Map your CSV column names to standard field types:

```json
"field_mappings": {
  "material": "Material_Column",
  "color_name": "Color_Description"
}
```

#### Standard Fields

- Product Attributes: `material`, `color_name`, `size`, `category`, `subcategory`
- Identifiers: `product_name`, `product_id`, `brand_name`
- Descriptions: `description`, `care_instructions`
- Business Fields: `price`, `country`, `gender`

## Detection Configuration

### Validation Rules

Validation rules are typically defined per validator (see validators module). Error messages are kept in JSON and loaded by reporters.

### Pattern Rules

Pattern-based detection rules live in `anomaly_detectors/pattern_based/rules/<field_name>.json` and follow this structure:

```json
{
  "field_name": "material",
  "description": "Pattern rules for material field",
  "known_values": ["cotton", "polyester", "wool", "silk"],
  "format_patterns": [
    {
      "name": "material_format",
      "pattern": "^[a-zA-Z\\s\\-%/]+$",
      "message": "Contains invalid characters",
      "probability": 0.8
    }
  ],
  "validation_rules": [
    { "name": "not_empty", "type": "not_empty", "message": "Value cannot be empty" },
    { "name": "max_length", "type": "max_length", "max_length": 100, "message": "Value is too long" },
    { "name": "min_length", "type": "min_length", "min_length": 1, "message": "Value is too short" }
  ]
}
```

## Environment Variables

Optional environment variables:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0
```

## Output Files

Single-sample demo output structure:

```
output_dir/
├── <sample>_viewer_report.json
├── <sample>_unified_report.json
├── <sample>_overall_confusion_matrix.png
├── <sample>_per_field_confusion_matrix.png
├── <sample>_detection_type_confusion_matrix.png
├── <sample>_performance_comparison.png
└── <sample>_summary_visualization.png
```

## Creating a New Brand Configuration

1. Copy an existing config and edit:
   ```bash
   cp brand_configs/esqualo.json brand_configs/new_brand.json
   ```
2. Edit field mappings for your dataset
3. Test the configuration:
   ```python
   from common.brand_config import load_brand_config
   config = load_brand_config('new_brand')
   print(config.get_column_name('material'))
   ```
4. Run detection:
   ```bash
   python main.py single-demo --data-file your_data.csv
   ```

## Best Practices

- Map only columns that exist in your data
- Use exact column names (case-sensitive)
- Start with core fields, add more gradually
- Start with default thresholds; adjust after evaluation
- Keep one config file per brand/dataset; version control them