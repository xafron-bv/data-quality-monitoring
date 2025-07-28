# Pattern-Based Anomaly Detection

Rule-based anomaly detection using configurable JSON patterns.

## Overview

This module provides a flexible, configuration-driven approach to anomaly detection. Instead of hardcoding detection logic, rules are defined in JSON files that are automatically loaded for each field.

## Key Features

- **No Training Required**: Works immediately with rule definitions
- **Field-Agnostic**: Same detector code works for all fields
- **Easy Configuration**: JSON-based rule definitions
- **Hot Reloading**: Changes to rules don't require code changes

## Rule Structure

Rules are stored in `rules/{field_name}.json`:

```json
{
  "field_name": "material",
  "description": "Pattern rules for material field",
  "known_values": [
    "cotton",
    "polyester", 
    "wool",
    "silk"
  ],
  "format_patterns": [
    {
      "name": "material_format",
      "pattern": "^[a-zA-Z\\s\\-%/]+$",
      "message": "Contains invalid characters",
      "probability": 0.8
    }
  ],
  "validation_rules": [
    {
      "name": "not_empty",
      "type": "not_empty",
      "message": "Material cannot be empty",
      "probability": 0.9
    },
    {
      "name": "max_length",
      "type": "max_length",
      "max_length": 100,
      "message": "Material description too long",
      "probability": 0.7
    }
  ]
}
```

## Rule Types

### 1. Known Values
- List of valid values (case-insensitive)
- Anything not in list is flagged as anomaly
- Best for fields with limited valid options

### 2. Format Patterns
- Regular expressions for format validation
- Can define multiple patterns
- Each pattern has configurable probability

### 3. Validation Rules
Built-in rule types:
- `not_empty`: Value must not be empty
- `max_length`: Maximum character length
- `min_length`: Minimum character length
- `numeric_range`: Value must be in numeric range

## Adding a New Field

1. Create rule file: `rules/{new_field}.json`
2. Define known values, patterns, and rules
3. The detector automatically loads and uses the rules

Example for a new "brand" field:
```json
{
  "field_name": "brand",
  "description": "Brand name validation",
  "known_values": [
    "nike",
    "adidas",
    "puma",
    "reebok"
  ],
  "format_patterns": [
    {
      "name": "brand_format",
      "pattern": "^[A-Za-z][A-Za-z\\s&-]*$",
      "message": "Invalid brand name format"
    }
  ],
  "validation_rules": [
    {
      "name": "not_empty",
      "type": "not_empty",
      "message": "Brand cannot be empty"
    }
  ]
}
```

## Advanced Patterns

### Complex Format Validation
```json
{
  "name": "ean_format",
  "pattern": "^[0-9]{13}$",
  "message": "EAN must be exactly 13 digits"
}
```

### Multiple Patterns (OR logic)
```json
"format_patterns": [
  {
    "name": "size_numeric",
    "pattern": "^[0-9]+$",
    "message": "Invalid numeric size"
  },
  {
    "name": "size_alpha",
    "pattern": "^(XS|S|M|L|XL|XXL)$",
    "message": "Invalid alpha size"
  }
]
```

## Configuration Options

### Probability Scores
Each rule can have a custom probability (0.0-1.0):
- Higher = more confident it's an anomaly
- Allows fine-tuning detection sensitivity

### Rule Priorities
Rules are evaluated in order:
1. Validation rules (structural checks)
2. Format patterns (format checks)
3. Known values (content checks)

## Best Practices

1. **Start Simple**: Begin with basic patterns and known values
2. **Iterate**: Refine rules based on false positives/negatives
3. **Document**: Use descriptive names and messages
4. **Test**: Validate rules with sample data

## Performance

- **Fast**: No model loading or inference
- **Lightweight**: Minimal memory usage
- **Scalable**: O(1) lookup for known values

## Troubleshooting

### Pattern Not Matching
- Test regex patterns separately
- Check for special character escaping
- Verify case sensitivity

### Too Many False Positives
- Expand known values list
- Adjust probability scores
- Make patterns less restrictive

### Rules Not Loading
- Check JSON syntax
- Verify file path and naming
- Look for error messages in console