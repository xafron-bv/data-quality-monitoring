# Validators Module

High-confidence, rule-based validation for data quality monitoring.

## Overview

Validators implement business rules and format constraints to detect definitive errors in data. Unlike anomaly detectors that identify unusual patterns, validators catch violations of known rules with high confidence.

## Structure

The validation system uses a JSON-based approach similar to pattern-based anomaly detectors:

- `rules/{field_name}/{variation}.json`: Validation rules and error messages for each field variation
- `json_validator.py`: Generic validator that reads rules from JSON files
- `json_reporter.py`: Reporter that formats error messages using JSON configuration

### Variation-Specific Rules (Required)

There are no default variations. Every field used by a brand must have a specified `variation` in `brand_configs/{brand}.json`:

- Place rules under `rules/{field_name}/{variation}.json`
- If `field_variations.{field_name}` is missing or the file does not exist, the system raises an error.

Example:

```
validators/
└── rules/
    └── season/
        ├── year_first.json        # e.g., "2025 Winter"
        └── name_first.json        # e.g., "Winter 2025"
```

In `brand_configs/mybrand.json`:

```
{
  "brand_name": "mybrand",
  "field_mappings": { "season": "season" },
  "field_variations": { "season": "year_first" }
}
```

## Creating a New Validator

### 1. Create a JSON Rules File

Create a new file in the `rules/{field}/` directory, named `{variation}.json`.

### 2. Define Validation Rules

```json
{
  "field_name": "new_field",
  "description": "Validation rules for new field",
  "error_messages": {
    "EMPTY_VALUE": "Field cannot be empty",
    "INVALID_FORMAT": "Invalid format: {value}",
    "OUT_OF_RANGE": "Value {value} is out of acceptable range"
  },
  "validation_rules": [
    { "name": "empty_check", "type": "empty_string", "error_code": "EMPTY_VALUE", "probability": 1.0 },
    { "name": "format_check", "type": "regex", "pattern": "^[A-Z][0-9]{3}$", "error_code": "INVALID_FORMAT", "probability": 0.95 }
  ]
}
```

## Supported Validation Types

- Basic: `missing`, `type_check`, `empty_string`, `whitespace`, `min_length`, `max_length`
- Pattern: `regex`, `regex_multiple`, `regex_negative`
- Content: `keyword_check`, `percentage_sum_check`, `parenthesis_check`, `year_range_check`, `temperature_check`
- Advanced: `contradiction_check`, `language_consistency`

## Available Validation Rules

Create per-variation files as needed under `rules/{field}/`.

## Error Injection

The `error_injection_rules/` directory contains JSON files defining how to inject errors for testing validators.

## Best Practices

1. **Explicit Variations**: Always set `field_variations` in brand config
2. **High Confidence**: Only flag definitive errors
3. **Clear Messages**: Provide actionable error descriptions
4. **Fast Execution**: Validators should be lightweight
5. **Consistent Interface**: Follow the ValidatorInterface contract