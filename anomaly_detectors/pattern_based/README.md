# Pattern-Based Anomaly Detection

## Overview

The pattern-based detection layer identifies anomalies using statistical analysis and domain-specific pattern matching. Unlike validators that enforce strict rules, pattern detectors flag statistical outliers and unusual patterns that may indicate data quality issues.

## Architecture

### Pattern-Based Detector

The `PatternBasedDetector` class provides a generic framework that loads field-specific rules from JSON configuration files:

```python
class PatternBasedDetector(AnomalyDetectorInterface):
    def __init__(self, field_name: str)
    def bulk_detect(self, df: pd.DataFrame, column_name: str) -> List[AnomalyError]
```

### Rule Configuration

Pattern rules are stored in `rules/{field_name}.json`:

```json
{
  "known_values": ["Cotton", "Polyester", "Wool"],
  "format_patterns": [
    "^\\d+%\\s+[A-Za-z]+$",
    "^[A-Za-z]+\\s+\\d+%$"
  ],
  "validation_rules": [
    {
      "type": "length_check",
      "min_length": 3,
      "max_length": 100
    }
  ]
}
```

## Detection Methods

### 1. Known Value Detection
- Compares values against a curated list of known valid values
- Flags values not in the known set as potential anomalies
- Useful for fields with limited valid options (e.g., seasons, categories)

### 2. Format Pattern Matching
- Uses regex patterns to validate expected formats
- Detects values that don't match any known patterns
- Supports multiple valid patterns per field

### 3. Statistical Anomaly Detection
- **Length Analysis**: Flags values with unusual lengths
- **Character Distribution**: Detects abnormal character usage
- **Frequency Analysis**: Identifies rare or unusual values
- **Similarity Clustering**: Groups similar values and flags outliers

### 4. Domain-Specific Rules
- **Material Composition**: Validates percentage sum and format
- **Category Hierarchy**: Checks category consistency
- **Temporal Patterns**: Validates season/date patterns
- **Language Consistency**: Detects mixed language usage

## Field-Specific Patterns

### Material Patterns
- **Valid Formats**: "100% Cotton", "50% Wool 50% Polyester"
- **Known Materials**: Cotton, Polyester, Wool, Silk, etc.
- **Composition Rules**: Percentages must sum to 100%
- **Suspicious Patterns**: Mixed formats, unusual delimiters

### Category Patterns
- **Hierarchical Validation**: Parent/child category relationships
- **Naming Conventions**: Standard category naming patterns
- **Length Constraints**: Reasonable category name lengths
- **Character Sets**: Allowed characters in categories

### Color Patterns
- **Standard Colors**: Common color names and variations
- **Format Patterns**: "Light Blue", "Blue/White", "RGB(255,0,0)"
- **Language Consistency**: Single language per value
- **Special Formats**: Hex codes, color codes

## Anomaly Scoring

Pattern detectors assign anomaly scores based on:

1. **Pattern Deviation**: How far value deviates from known patterns
2. **Rarity**: How uncommon the value is in the dataset
3. **Rule Violations**: Number and severity of rule violations
4. **Confidence Weights**: Pre-configured weights for each rule type

```python
score = (
    pattern_score * 0.4 +
    rarity_score * 0.3 +
    rule_violation_score * 0.3
)
```

## Error Codes

Common pattern-based error codes:
- `UNKNOWN_VALUE`: Value not in known valid set
- `INVALID_FORMAT`: Doesn't match any format pattern
- `SUSPICIOUS_PATTERN`: Matches suspicious pattern
- `DOMAIN_VIOLATION`: Violates domain-specific rules

## Performance Optimization

- **Compiled Regex**: Patterns compiled once at initialization
- **Cached Results**: Common patterns cached during detection
- **Batch Processing**: Vectorized operations for efficiency
- **Early Termination**: Stops checking once threshold exceeded

## Extending Pattern Detection

To add patterns for a new field:

1. Create `rules/{field_name}.json`
2. Define:
   - `known_values`: List of valid values
   - `format_patterns`: Regex patterns for valid formats
   - `validation_rules`: Custom validation logic
3. Pattern detector automatically loads and applies rules

## Integration with Other Detectors

Pattern detection works alongside:
- **Validators**: Pattern detection is probabilistic, validation is deterministic
- **ML Detection**: Patterns provide rule-based baseline, ML captures complex anomalies
- **LLM Detection**: Patterns offer interpretable rules, LLMs provide context understanding

## Best Practices

1. **Balance Sensitivity**: Avoid too many false positives
2. **Domain Knowledge**: Encode field-specific expertise in rules
3. **Regular Updates**: Keep known values and patterns current
4. **Clear Explanations**: Provide clear reasons for anomaly flags
5. **Performance Testing**: Profile pattern matching on large datasets
