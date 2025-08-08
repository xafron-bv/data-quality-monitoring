# Validators Module

High-confidence, rule-based validation for data quality monitoring.

## Overview

Validators implement business rules and format constraints to detect definitive errors in data. Unlike anomaly detectors that identify unusual patterns, validators catch violations of known rules with high confidence.

## Two Ways to Author Validation

- Field-specific Python validators in `validators/{field}/validate.py` (legacy/custom)
- Generic rule-based validator with JSON rules in `validators/rule_based/rules/{field}.json` (recommended)

The rule-based approach mirrors the pattern-based anomaly detection: one JSON per field, hot-reloaded without code changes.

## Rule-Based Validator

- Code: `validators/rule_based/rule_based_validator.py`
- Rules: `validators/rule_based/rules/{field}.json`
- Reporter: `validators/report.py` (falls back to generic messages)

Example rule file:

```json
{
  "field_name": "material",
  "description": "Validation rules for material",
  "known_values": ["cotton", "polyester"],
  "format_patterns": [
    {"name": "chars", "pattern": "^[A-Za-z0-9\\s%/().,-]+$", "message": "Invalid characters"}
  ],
  "validation_rules": [
    {"name": "not_empty", "type": "not_empty", "message": "Value cannot be empty"},
    {"name": "maxlen", "type": "max_length", "max_length": 120, "message": "Too long"},
    {"name": "must_contain_pct", "type": "must_contain", "substring": "%", "message": "Missing %"}
  ]
}
```

Supported rule types:
- not_empty
- max_length (max_length)
- min_length (min_length)
- numeric_range (min, max)
- must_contain (substring)
- must_not_contain (substring)
- regex_must_match (pattern)
- regex_must_not_match (pattern)

## Creating a New Validator (Rule-Based)

1. Create `validators/rule_based/rules/{field}.json`. On first run, a template will be generated automatically if missing.
2. Populate `known_values`, `format_patterns`, and `validation_rules` as needed.
3. Use any pipeline that enables validation; the system will load the rules and validate.

## Creating a New Validator (Custom Python)

Each field validator consists of:
- `{field_name}/validate.py`: Validator implementation
- `{field_name}/error_messages.json`: Human-readable error descriptions

```python
# validators/new_field/validate.py
from validators.validator_interface import ValidatorInterface
from validators.validation_error import ValidationError

class Validator(ValidatorInterface):
    def _validate_entry(self, value):
        if not value or str(value).strip() == "":
            return ValidationError(error_type="EMPTY_VALUE", probability=1.0, details={})
        return None
```

Error messages file example:

```json
{
  "EMPTY_VALUE": "Value cannot be empty"
}
```

## Available Validators

- `rule_based/`: Generic validator powered by JSON rules
- `care_instructions/`: Custom validator
- `category/`: Custom validator
- `color_name/`: Custom validator
- `material/`: Custom validator
- `season/`: Custom validator
- `size/`: Custom validator
- `template/`: Template for new custom validators

## Best Practices

1. **High Confidence**: Only flag definitive errors
2. **Clear Messages**: Provide actionable error descriptions
3. **Fast Execution**: Validators should be lightweight
4. **Consistent Interface**: Follow the ValidatorInterface contract