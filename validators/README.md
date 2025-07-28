# Validators Module

High-confidence, rule-based validation for data quality monitoring.

## Overview

Validators implement business rules and format constraints to detect definitive errors in data. Unlike anomaly detectors that identify unusual patterns, validators catch violations of known rules with high confidence.

## Structure

Each field validator consists of:
- `{field_name}/validate.py`: Validator implementation
- `{field_name}/error_messages.json`: Human-readable error descriptions

## Creating a New Validator

### 1. Create Directory Structure
```bash
validators/
└── new_field/
    ├── validate.py
    └── error_messages.json
```

### 2. Implement Validator Class

```python
# validators/new_field/validate.py
from validators.validator_interface import ValidatorInterface
from validators.validation_error import ValidationError

class Validator(ValidatorInterface):
    def __init__(self):
        self.field_type = "new_field"
    
    def validate(self, value, row_index=None):
        errors = []
        
        # Add validation logic
        if not value or str(value).strip() == "":
            errors.append(ValidationError(
                error_type="EMPTY_VALUE",
                severity="ERROR",
                confidence=1.0,
                details={"value": value}
            ))
        
        return errors
```

### 3. Define Error Messages

```json
{
  "EMPTY_VALUE": {
    "message": "Value cannot be empty",
    "description": "This field requires a non-empty value",
    "severity": "ERROR",
    "examples": ["", " ", null]
  }
}
```

## Common Validation Patterns

### Format Validation
- Regular expressions for patterns
- Length constraints
- Character set restrictions

### Business Rules
- Required fields
- Value dependencies
- Cross-field validation

### Domain Constraints
- Allowed value lists
- Range validation
- Logical constraints

## Available Validators

- `care_instructions/`: Validates care instruction formats
- `category/`: Product category validation
- `color_name/`: Color name validation
- `material/`: Material composition validation
- `season/`: Season code validation
- `size/`: Size format validation
- `template/`: Generic template for new validators

## Error Injection

The `error_injection_rules/` directory contains JSON files defining how to inject errors for testing validators.

## Best Practices

1. **High Confidence**: Only flag definitive errors
2. **Clear Messages**: Provide actionable error descriptions
3. **Fast Execution**: Validators should be lightweight
4. **Consistent Interface**: Follow the ValidatorInterface contract