# Validation Methods Documentation

## Overview

The validation layer provides field-specific, rule-based data quality checks. Each field type has a dedicated validator that implements the `ValidatorInterface` and performs deterministic validation based on business rules and data format requirements.

## Validator Interface

All validators implement the standard interface:

```python
class ValidatorInterface(ABC):
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]
    def _validate_entry(self, value: Any) -> Optional[ValidationError]
```

## Field-Specific Validators

### Material Validator

**Purpose**: Validates material composition strings (e.g., "100% Cotton", "50% Wool 50% Polyester")

**Key Validation Rules**:
- **Missing/Empty Values**: Flags missing or empty material specifications
- **Format Compliance**: Ensures proper percentage format with material names
- **Composition Sum**: Verifies percentages sum to 100%
- **Invalid Characters**: Detects special characters not allowed in material specs
- **Whitespace Issues**: Identifies extraneous whitespace or line breaks
- **Duplicate Materials**: Catches repeated material names in composition
- **Malformed Tokens**: Identifies improperly formatted percentage/material pairs

**Error Codes**:
- `MISSING_VALUE`: No material data provided
- `SUM_NOT_100`: Material percentages don't sum to 100%
- `INVALID_CHARACTERS`: Contains disallowed characters
- `MALFORMED_TOKEN`: Improper percentage/material format
- `DUPLICATE_MATERIAL_NAME`: Same material listed multiple times

### Category Validator

**Purpose**: Validates product category values

**Key Validation Rules**:
- **Type Validation**: Ensures category is a string
- **Empty Categories**: Flags empty category values
- **Whitespace Errors**: Detects leading/trailing spaces
- **Special Characters**: Identifies invalid characters in categories
- **HTML Tags**: Catches embedded HTML in category names
- **Random Noise**: Detects gibberish or malformed text

**Error Codes**:
- `MISSING_VALUE`: No category provided
- `INVALID_TYPE`: Non-string category value
- `WHITESPACE_ERROR`: Improper whitespace formatting
- `SPECIAL_CHARACTERS`: Contains invalid characters
- `HTML_TAGS`: HTML markup detected

### Color Name Validator

**Purpose**: Validates color name specifications

**Key Validation Rules**:
- **Format Validation**: Ensures proper color name format
- **Special Characters**: Detects invalid characters in color names
- **Whitespace Issues**: Identifies formatting problems
- **Length Constraints**: Validates reasonable color name lengths
- **Known Patterns**: Checks against common color naming patterns

### Size Validator

**Purpose**: Validates size specifications (numeric and alphanumeric)

**Key Validation Rules**:
- **Format Patterns**: Supports XS/S/M/L/XL, numeric (36-52), and custom formats
- **Range Validation**: Ensures numeric sizes are within reasonable ranges
- **Consistency**: Validates size format consistency
- **Special Formats**: Handles size ranges (e.g., "36-38")

### Care Instructions Validator

**Purpose**: Validates care instruction text

**Key Validation Rules**:
- **Language Detection**: Ensures instructions are in expected language
- **Symbol Validation**: Checks for proper care symbols
- **Completeness**: Validates all required care aspects are covered
- **Format Compliance**: Ensures standard care instruction format

### Season Validator

**Purpose**: Validates season specifications

**Key Validation Rules**:
- **Valid Seasons**: Checks against allowed season values
- **Format Validation**: Ensures proper season/year format
- **Temporal Logic**: Validates season makes sense for product

## Validation Error Structure

All validators return `ValidationError` objects with:

```python
@dataclass
class ValidationError:
    error_type: str           # Error code enum value
    probability: float        # Confidence score (0.0-1.0)
    details: Dict[str, Any]   # Additional context
    row_index: Optional[int]  # DataFrame row index
    error_data: Optional[Any] # The invalid value
```

## Confidence Scoring

Validators assign confidence scores based on error severity:
- **1.0**: Definite errors (missing values, type mismatches)
- **0.9-0.99**: Very likely errors (invalid formats, constraint violations)
- **0.8-0.89**: Probable errors (suspicious patterns)
- **0.6-0.79**: Possible errors (ambiguous cases)

## Error Injection Rules

Each validator has corresponding error injection rules in `error_injection_rules/`:

```json
{
  "error_type": "MISSING_VALUE",
  "operation": "replace",
  "target": "original_value",
  "replacement": null,
  "probability": 1.0
}
```

## Performance Optimization

- **Vectorized Operations**: Bulk validation uses pandas operations where possible
- **Regex Compilation**: Patterns compiled once and reused
- **Early Exit**: Validation stops at first error for efficiency
- **Caching**: Common validation results cached during bulk operations

## Extending Validators

To add a new field validator:

1. Create `validators/<field_name>/validate.py`
2. Implement `ValidatorInterface`
3. Define field-specific `ErrorCode` enum
4. Implement `_validate_entry` method
5. Add error injection rules in `error_injection_rules/<field_name>.json`

## Best Practices

1. **Deterministic Rules**: Validators should be deterministic and rule-based
2. **Clear Error Messages**: Provide actionable error descriptions
3. **Appropriate Confidence**: Set confidence scores based on certainty
4. **Performance Focus**: Optimize for bulk validation scenarios
5. **Extensibility**: Design for easy addition of new validation rules