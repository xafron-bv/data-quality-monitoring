# Rule-Based Validation

Rule-based validation is the first line of defense in the anomaly detection system, providing fast and deterministic checks against predefined rules.

## Overview

Rule-based validation uses field-specific validators that implement business logic and domain constraints. Each validator is designed for a specific field type and enforces rules appropriate to that domain.

## Architecture

```
Input Data → Field Validator → Validation Rules → Error/Success
                   ↓                   ↓
              Field Type          Rule Engine
```

## Available Validators

### Care Instructions Validator

Validates care instruction text and symbols:

- **Washing symbols**: Machine wash, hand wash, do not wash
- **Temperature ranges**: Cold, warm, hot
- **Drying methods**: Tumble dry, line dry, flat dry
- **Special instructions**: Dry clean only, do not bleach

### Category Validator

Validates product category hierarchies:

- **Category structure**: Parent → Child → Sub-child
- **Valid paths**: Clothing → Tops → T-Shirts
- **Naming conventions**: Consistent capitalization
- **Depth limits**: Maximum hierarchy levels

### Color Name Validator

Validates color naming and descriptions:

- **Standard colors**: Red, Blue, Green, etc.
- **Compound colors**: Navy Blue, Forest Green
- **Pattern descriptions**: Striped, Polka Dot
- **Brand-specific colors**: Custom color palettes

### Material Validator

Validates material composition:

- **Material names**: Cotton, Polyester, Wool
- **Percentage validation**: Must sum to 100%
- **Format checking**: "100% Cotton" vs "Cotton 100%"
- **Blend validation**: Valid material combinations

### Season Validator

Validates seasonal classifications:

- **Season names**: Spring, Summer, Fall, Winter
- **Year validation**: SS23, FW24
- **Collection types**: Resort, Pre-Fall
- **Date ranges**: Season-appropriate dates

### Size Validator

Validates size specifications:

- **Standard sizes**: XS, S, M, L, XL, XXL
- **Numeric sizes**: 0-16, 28-42
- **International conversions**: US/UK/EU sizes
- **Size ranges**: Valid min/max values

## Implementation Pattern

All validators follow the `ValidatorInterface`:

```python
class ValidatorInterface(ABC):
    @abstractmethod
    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """Validate a single entry"""
        pass
    
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """Validate an entire column"""
        # Implementation provided by base class
```

## Rule Types

### Format Rules

Check data format and structure:

```python
# Example: Email format
pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
if not re.match(pattern, value):
    return ValidationError("Invalid email format")
```

### Domain Rules

Enforce business logic:

```python
# Example: Price validation
if price < 0:
    return ValidationError("Price cannot be negative")
if price > max_price_limit:
    return ValidationError("Price exceeds maximum limit")
```

### Referential Rules

Check relationships between fields:

```python
# Example: Size consistency
if category == "Shoes" and size_type == "Letter":
    return ValidationError("Shoes should use numeric sizes")
```

### Temporal Rules

Validate time-based constraints:

```python
# Example: Season validation
if season == "Spring/Summer" and month in [10, 11, 12]:
    return ValidationError("Spring/Summer items in winter months")
```

## Configuration

### Rule Definition

Rules can be defined in JSON configuration:

```json
{
  "field_name": "color",
  "rules": [
    {
      "type": "format",
      "pattern": "^[A-Za-z\\s]+$",
      "message": "Color must contain only letters and spaces"
    },
    {
      "type": "domain",
      "valid_values": ["Red", "Blue", "Green", "Yellow"],
      "message": "Unknown color"
    }
  ]
}
```

### Threshold Configuration

```json
{
  "validators": {
    "strict_mode": true,
    "error_threshold": 0.05,
    "warning_threshold": 0.10
  }
}
```

## Error Reporting

### ValidationError Structure

```python
class ValidationError:
    def __init__(self, 
                 error_type: str,
                 message: str,
                 severity: str = "ERROR"):
        self.error_type = error_type
        self.message = message
        self.severity = severity
        self.row_index = None
        self.column_name = None
        self.value = None
```

### Error Types

- **FORMAT_ERROR**: Data doesn't match expected format
- **DOMAIN_ERROR**: Value outside valid domain
- **MISSING_REQUIRED**: Required field is empty
- **INVALID_TYPE**: Wrong data type
- **CONSTRAINT_VIOLATION**: Business rule violated

## Performance Optimization

### Bulk Validation

Process entire columns efficiently:

```python
def bulk_validate(self, df: pd.DataFrame, column_name: str):
    # Vectorized operations where possible
    mask = df[column_name].str.match(pattern)
    errors = df[~mask].apply(create_error)
```

### Early Exit

Stop processing on critical errors:

```python
if critical_error_found:
    return errors  # Don't continue validation
```

### Caching

Cache compiled patterns and lookups:

```python
@lru_cache(maxsize=100)
def get_compiled_pattern(pattern: str):
    return re.compile(pattern)
```

## Integration

### With Other Detection Methods

```python
# Run validation first
validation_errors = validator.bulk_validate(df, column)

# Only run expensive detection if validation passes
if len(validation_errors) < threshold:
    anomalies = ml_detector.detect(df, column)
```

### Custom Validators

Create new validators by extending the interface:

```python
class CustomValidator(ValidatorInterface):
    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        # Custom validation logic
        if not meets_custom_criteria(value):
            return ValidationError(
                error_type="CUSTOM_ERROR",
                message="Value fails custom criteria"
            )
        return None
```

## Best Practices

1. **Define Clear Rules**: Document what constitutes valid data
2. **Use Appropriate Severity**: ERROR vs WARNING vs INFO
3. **Provide Helpful Messages**: Explain why validation failed
4. **Consider Performance**: Use vectorized operations
5. **Handle Edge Cases**: Null values, empty strings, special characters
6. **Test Thoroughly**: Unit tests for each rule
7. **Monitor False Positives**: Track and tune rules

## Examples

### Material Validation

```python
def validate_material(value: str) -> Optional[ValidationError]:
    # Check format
    if not re.match(r'^\d+%\s+\w+', value):
        return ValidationError("Invalid material format")
    
    # Parse percentages
    percentages = extract_percentages(value)
    if sum(percentages) != 100:
        return ValidationError("Percentages must sum to 100%")
    
    # Check material names
    materials = extract_materials(value)
    for material in materials:
        if material not in VALID_MATERIALS:
            return ValidationError(f"Unknown material: {material}")
    
    return None
```

### Size Validation

```python
def validate_size(value: str, category: str) -> Optional[ValidationError]:
    if category == "Clothing":
        if value not in ["XS", "S", "M", "L", "XL", "XXL"]:
            return ValidationError("Invalid clothing size")
    elif category == "Shoes":
        try:
            size = float(value)
            if size < 5 or size > 15:
                return ValidationError("Shoe size out of range")
        except ValueError:
            return ValidationError("Shoe size must be numeric")
    
    return None
```