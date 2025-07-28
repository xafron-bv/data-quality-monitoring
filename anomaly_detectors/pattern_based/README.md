# Pattern-Based Anomaly Detection

Statistical pattern-based approach for detecting anomalies through analysis of data patterns, distributions, and outliers.

## Architecture Overview

Pattern-based detectors analyze statistical properties and patterns within data to identify anomalies without requiring machine learning models or training data.

### Key Principles
- **Statistical Analysis**: Use statistical methods to identify outliers
- **Pattern Recognition**: Detect deviations from expected patterns
- **Field-Specific Logic**: Custom detection logic per data type
- **Deterministic**: Reproducible results without randomness

## System Structure

```
pattern_based/
├── pattern_based_detector.py    # Factory for creating field-specific detectors
├── material/
│   └── detect.py               # Material-specific pattern detection
├── color_name/
│   └── detect.py               # Color name pattern detection
├── size/
│   └── detect.py               # Size pattern detection
├── category/
│   └── detect.py               # Category pattern detection
├── care_instructions/
│   └── detect.py               # Care instructions pattern detection
└── [other fields]/
    └── detect.py               # Field-specific detectors
```

## Detection Approaches

### 1. Text Length Analysis
Identifies values with unusual lengths compared to the field's distribution:

```python
def detect_length_anomalies(values):
    lengths = [len(str(v)) for v in values]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    
    # Flag values > 2 standard deviations from mean
    anomalies = []
    for value, length in zip(values, lengths):
        z_score = (length - mean_len) / std_len
        if abs(z_score) > 2:
            anomalies.append(value)
    return anomalies
```

### 2. Character Pattern Analysis
Detects unusual character distributions or invalid characters:

- Special character frequency
- Numeric/alphabetic ratios
- Unicode anomalies
- Repeated character patterns

### 3. Statistical Outlier Detection
Uses various statistical methods:

- **Z-score**: Distance from mean in standard deviations
- **IQR**: Interquartile range for robust outlier detection
- **Isolation Forest**: For multivariate outlier detection
- **Frequency-based**: Rare values in categorical fields

### 4. Format Validation
Checks structural patterns:

- Expected delimiters
- Component counts
- Pattern matching (regex)
- Format consistency

## Field-Specific Implementations

### Material Detector
- Composition percentage validation
- Component count analysis
- Delimiter consistency
- Text pattern matching

### Color Name Detector
- Length distribution analysis
- Common color term validation
- Special character detection
- Case consistency

### Size Detector
- Size format validation
- Numeric range checking
- Standard size detection
- Outlier identification

### Category Detector
- Hierarchical structure validation
- Path depth analysis
- Delimiter consistency
- Term frequency analysis

## Usage

### Direct Usage

```python
from anomaly_detectors.pattern_based.material.detect import MaterialAnomalyDetector

# Create detector
detector = MaterialAnomalyDetector()

# Detect anomalies
anomalies = detector.bulk_detect(df, "material")

for anomaly in anomalies:
    print(f"Row {anomaly.row_index}: {anomaly.anomaly_type}")
    print(f"Confidence: {anomaly.probability}")
    print(f"Details: {anomaly.details}")
```

### Factory Pattern

```python
from anomaly_detectors.pattern_based.pattern_based_detector import PatternBasedDetectorFactory

# Create appropriate detector for field
detector = PatternBasedDetectorFactory.create_detector("material")

# Use detector
anomalies = detector.bulk_detect(df, column_name)
```

## Detection Confidence

Pattern-based detectors assign confidence scores based on:

- **High (0.7-0.9)**: Clear statistical outliers
- **Medium (0.5-0.7)**: Moderate deviations
- **Low (0.3-0.5)**: Slight anomalies

Confidence calculation factors:
- Severity of deviation
- Multiple anomaly indicators
- Field-specific heuristics

## Performance Characteristics

### Strengths
- **No training required**: Works immediately on new data
- **Interpretable**: Clear statistical reasoning
- **Fast**: Efficient statistical computations
- **Deterministic**: Consistent results

### Limitations
- **Context-blind**: Doesn't understand semantic meaning
- **Rule-based**: May miss complex patterns
- **Field-specific**: Requires custom implementation per field
- **Statistical assumptions**: May not fit all distributions

## Integration

Pattern-based detectors implement the standard `AnomalyDetectorInterface`:

```python
class AnomalyDetectorInterface(ABC):
    @abstractmethod
    def _detect_anomaly(self, value, context=None):
        """Detect anomaly in single value"""
        pass
    
    def bulk_detect(self, df, column_name, batch_size=None):
        """Detect anomalies in DataFrame column"""
        pass
```

## Extending Pattern Detection

### Adding New Field Detector

1. Create new directory: `pattern_based/new_field/`
2. Implement detector in `detect.py`:

```python
class NewFieldAnomalyDetector(AnomalyDetectorInterface):
    def _detect_anomaly(self, value, context=None):
        # Custom detection logic
        if is_anomalous(value):
            return AnomalyError(
                anomaly_type="pattern_deviation",
                probability=0.8,
                details={"reason": "explanation"}
            )
        return None
```

3. Register in factory pattern
4. Add field-specific detection logic

### Best Practices
- Combine multiple detection methods
- Use robust statistics (median vs mean)
- Consider field semantics
- Validate with real anomaly examples
- Document detection logic clearly