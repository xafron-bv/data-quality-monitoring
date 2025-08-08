# Validators Module

High-confidence, rule-based validation for data quality monitoring.

## Overview

Validators implement business rules and format constraints to detect definitive errors in data. Unlike anomaly detectors that identify unusual patterns, validators catch violations of known rules with high confidence.

## Structure

The validation system uses a JSON-based approach similar to pattern-based anomaly detectors:

- `rules/{field_name}.json`: Validation rules and error messages for each field
- `json_validator.py`: Generic validator that reads rules from JSON files
- `json_reporter.py`: Reporter that formats error messages using JSON configuration

## Creating a New Validator

### 1. Create a JSON Rules File

Create a new file in the `rules/` directory:

```bash
validators/
└── rules/
    └── new_field.json
```

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
    {
      "name": "empty_check",
      "description": "Check for empty values",
      "type": "empty_string",
      "error_code": "EMPTY_VALUE",
      "probability": 1.0
    },
    {
      "name": "format_check",
      "description": "Check format using regex",
      "type": "regex",
      "pattern": "^[A-Z][0-9]{3}$",
      "error_code": "INVALID_FORMAT",
      "probability": 0.95
    }
  ]
}
```

## Supported Validation Types

The JSON validator supports the following validation types:

### Basic Validations
- `missing`: Check for missing/null values
- `type_check`: Validate data type (e.g., string, number)
- `empty_string`: Check for empty strings
- `whitespace`: Check for leading/trailing whitespace
- `min_length`: Minimum string length validation
- `max_length`: Maximum string length validation

### Pattern Matching
- `regex`: Single regex pattern matching
- `regex_multiple`: Multiple regex patterns (OR logic)
- `regex_negative`: Pattern should NOT match (valid if no match)

### Content Validation
- `keyword_check`: Ensure required keywords are present
- `percentage_sum_check`: Validate percentages sum to 100
- `parenthesis_check`: Check for balanced parentheses
- `year_range_check`: Validate years within range
- `temperature_check`: Validate temperature values

### Advanced Validations
- `contradiction_check`: Check for contradictory statements
- `language_consistency`: Check for mixed languages
- Custom validation functions can be added to the JSONValidator class

## Available Validation Rules

The following fields have JSON validation rules defined:

- `rules/care_instructions.json`: Care instruction validation
- `rules/category.json`: Product category validation
- `rules/color_name.json`: Color name validation
- `rules/material.json`: Material composition validation
- `rules/season.json`: Season validation
- `rules/size.json`: Size format validation

## Error Injection

The `error_injection_rules/` directory contains JSON files defining how to inject errors for testing validators.

## Best Practices

1. **High Confidence**: Only flag definitive errors
2. **Clear Messages**: Provide actionable error descriptions
3. **Fast Execution**: Validators should be lightweight
4. **Consistent Interface**: Follow the ValidatorInterface contract