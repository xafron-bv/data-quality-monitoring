# Rule-Based Validators

High-confidence, deterministic validation system for detecting data quality issues based on business rules and format requirements.

## Architecture Overview

Validators implement strict rule-based checks to identify definitive data quality issues with high confidence. Unlike anomaly detectors, validators flag violations of explicit business rules and format requirements.

### Key Principles
- **High Confidence**: Only flag definitive errors (0.8-1.0 probability)
- **Business Rules**: Enforce domain-specific requirements
- **Format Validation**: Check structural correctness
- **Deterministic**: Consistent, reproducible results

## System Structure

```
validators/
├── validator_interface.py       # Base interface for all validators
├── reporter_interface.py        # Base interface for reporters
├── material/
│   ├── validate.py             # Material composition validation
│   ├── report.py               # Material-specific reporting
│   └── error_messages.json     # Error code definitions
├── color_name/
│   ├── validate.py             # Color name validation
│   ├── report.py               # Color-specific reporting
│   └── error_messages.json     # Error code definitions
├── size/
│   ├── validate.py             # Size validation
│   ├── report.py               # Size-specific reporting
│   └── error_messages.json     # Error code definitions
└── [other fields]/
    ├── validate.py             # Field-specific validation
    ├── report.py               # Field-specific reporting
    └── error_messages.json     # Error code definitions
```

## Validation Architecture

### Base Interface

```python
class ValidatorInterface(ABC):
    @abstractmethod
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """Validate all values in a DataFrame column"""
        pass
```

### Validation Error Structure

```python
class ValidationError:
    def __init__(self, error_type: str, probability: float, details: Dict[str, Any]):
        self.error_type = error_type      # Error code
        self.probability = probability    # Confidence (0.8-1.0)
        self.details = details           # Additional context
        self.row_index = None           # Set by bulk_validate
        self.column_name = None         # Set by bulk_validate
        self.error_data = None          # Original value
```

## Field-Specific Validators

### Material Validator
Validates material composition data:

- **Percentage validation**: Sum must equal 100%
- **Format validation**: Expected structure (e.g., "95% Cotton - 5% Spandex")
- **Component validation**: Valid material names
- **Multi-part validation**: Handle complex compositions
- **Invalid patterns**: Detect corruption or nonsense

### Color Name Validator
Validates color descriptions:

- **Empty/null checks**: Missing color names
- **Invalid characters**: Special symbols, numbers
- **Format validation**: Expected patterns
- **Length validation**: Reasonable bounds
- **Business rules**: Brand-specific requirements

### Size Validator
Validates size information:

- **Format validation**: Standard size formats
- **Range validation**: Valid size ranges
- **Type consistency**: Numeric vs alphabetic
- **Missing values**: Required field checks
- **Cross-field validation**: Size-category compatibility

### Category Validator
Validates product categories:

- **Hierarchy validation**: Valid category paths
- **Delimiter consistency**: Proper separators
- **Depth validation**: Maximum nesting levels
- **Term validation**: Allowed category values
- **Completeness**: Required hierarchy levels

## Error Codes and Messages

Each validator maintains an `error_messages.json` file:

```json
{
  "MISSING_PERCENTAGE": {
    "message": "Material composition missing percentage values",
    "severity": "high",
    "probability": 0.95
  },
  "INVALID_TOTAL": {
    "message": "Material percentages do not sum to 100%",
    "severity": "high",
    "probability": 1.0
  },
  "INVALID_FORMAT": {
    "message": "Invalid material format",
    "severity": "medium",
    "probability": 0.85
  }
}
```

## Usage

### Basic Validation

```python
from validators.material.validate import MaterialValidator

# Create validator
validator = MaterialValidator()

# Validate data
errors = validator.bulk_validate(df, "material")

# Process errors
for error in errors:
    print(f"Row {error.row_index}: {error.error_type}")
    print(f"Value: {error.error_data}")
    print(f"Details: {error.details}")
```

### With Reporting

```python
from validators.material.validate import MaterialValidator
from validators.material.report import MaterialReporter

# Validate
validator = MaterialValidator()
errors = validator.bulk_validate(df, "material")

# Generate reports
reporter = MaterialReporter()
reports = reporter.generate_report(errors, df)

# Display reports
for report in reports:
    print(f"Issue: {report['display_message']}")
    print(f"Location: Row {report['row_index']}")
```

## Creating New Validators

### 1. Directory Structure
```
validators/new_field/
├── validate.py          # Validator implementation
├── report.py            # Reporter implementation
└── error_messages.json  # Error definitions
```

### 2. Implement Validator

```python
from validators.validator_interface import ValidatorInterface
from validators.validation_error import ValidationError

class NewFieldValidator(ValidatorInterface):
    def bulk_validate(self, df, column_name):
        errors = []
        
        for idx, value in df[column_name].items():
            # Validation logic
            if is_invalid(value):
                error = ValidationError(
                    error_type="INVALID_FORMAT",
                    probability=0.9,
                    details={"reason": "specific issue"}
                )
                error.row_index = idx
                error.column_name = column_name
                error.error_data = value
                errors.append(error)
        
        return errors
```

### 3. Implement Reporter

```python
from validators.reporter_interface import ReporterInterface

class NewFieldReporter(ReporterInterface):
    def generate_report(self, errors, df):
        reports = []
        
        for error in errors:
            report = {
                "row_index": error.row_index,
                "field_name": error.column_name,
                "value": error.error_data,
                "error_code": error.error_type,
                "display_message": self.get_message(error.error_type),
                "probability": error.probability,
                "details": error.details
            }
            reports.append(report)
        
        return reports
```

## Best Practices

### Validation Design
- **High confidence only**: Only flag definitive errors
- **Clear error messages**: User-friendly explanations
- **Detailed context**: Include relevant details
- **Performance**: Optimize for large datasets
- **Maintainability**: Clear, documented logic

### Error Probability Guidelines
- **1.0**: Absolute violations (math errors, impossible values)
- **0.95**: Near-certain errors (severe format violations)
- **0.90**: Very likely errors (missing required data)
- **0.85**: Probable errors (suspicious patterns)
- **0.80**: Minimum threshold for validation errors

## Integration

Validators integrate with the unified detection system:

- Higher priority than anomaly detectors
- Results always included regardless of thresholds
- Compatible with weighted combination system
- Provides structured error information

## Performance Considerations

- **Vectorized operations**: Use pandas operations where possible
- **Early termination**: Stop checking after finding errors
- **Batch processing**: Process in chunks for memory efficiency
- **Regex compilation**: Pre-compile patterns
- **Caching**: Cache expensive computations