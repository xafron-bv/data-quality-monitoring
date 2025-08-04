# Core Components

The anomaly detection system is built around several core components that work together to provide comprehensive data quality validation and anomaly detection capabilities.

## Component Overview

The system consists of the following main components:

1. **Validators** - Rule-based validation for specific field types
2. **Anomaly Detectors** - Pattern and ML-based anomaly detection
3. **Reporters** - Result formatting and reporting interfaces
4. **Common Utilities** - Shared functionality across components
5. **Data Processing Pipeline** - Orchestration and execution

## Validators

### ValidatorInterface

The `ValidatorInterface` is the abstract base class for all validators in the system. It provides a consistent interface for validation operations.

```python
class ValidatorInterface(ABC):
    @abstractmethod
    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """Validates a single data entry"""
        pass
    
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """Validates an entire column"""
        pass
```

### Available Validators

The system includes field-specific validators for:

- **Care Instructions** - Validates care instruction text and symbols
- **Category** - Validates product category hierarchies
- **Color Name** - Validates color naming conventions
- **Material** - Validates material composition and descriptions
- **Season** - Validates seasonal classifications
- **Size** - Validates size specifications and formats

Each validator implements the `ValidatorInterface` and provides field-specific validation logic.

## Anomaly Detectors

### AnomalyDetectorInterface

The `AnomalyDetectorInterface` defines the contract for all anomaly detection implementations:

```python
class AnomalyDetectorInterface(ABC):
    @abstractmethod
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """Detects anomalies in a single data entry"""
        pass
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """Learns patterns from training data (optional)"""
        pass
    
    def bulk_detect(self, df: pd.DataFrame, column_name: str, batch_size: Optional[int], max_workers: int) -> List[AnomalyError]:
        """Detects anomalies in bulk with parallel processing"""
        pass
```

### Detection Methods

The system supports multiple anomaly detection approaches:

1. **Pattern-Based Detection** - Uses predefined patterns and rules
2. **ML-Based Detection** - Uses machine learning models with GPU acceleration support
3. **LLM-Based Detection** - Leverages language models for complex text analysis

### Parallel Processing

The anomaly detection system includes optimized parallel processing:

- CPU-based detectors use multiprocessing for row-by-row processing
- GPU-based ML detectors process data in optimized batches
- Automatic batch size optimization based on available resources

## Reporters

### ReporterInterface

Reporters transform raw detection results into human-readable formats:

```python
class AnomalyReporterInterface(ABC):
    @abstractmethod
    def generate_report(self, 
                       anomaly_results: Union[List[AnomalyError], List[MLAnomalyResult]], 
                       original_df: pd.DataFrame,
                       threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generates human-readable reports from anomaly results"""
        pass
```

### Result Types

The system supports two types of results:

1. **ValidationError** - For rule-based validation failures
2. **AnomalyError** - For pattern/ML-based anomaly detections
3. **MLAnomalyResult** - Extended results with ML-specific information:
   - Probability scores
   - Feature contributions
   - Nearest neighbors
   - Clustering information
   - Model explanations

## Common Utilities

### Brand Configuration

The `BrandConfig` class manages brand-specific settings and rules:

- Field mappings
- Validation rules
- Detection thresholds
- Model configurations

### Field Mapping

The `FieldMapper` provides flexible field name mapping between different data sources:

- Column name normalization
- Brand-specific field naming
- Multi-language support

### Error Injection

For testing and evaluation, the system includes error injection capabilities:

- Systematic error introduction
- Anomaly pattern injection
- Configurable error rates and types

### Debug Configuration

The debug module provides runtime debugging controls:

- Debug print functions
- Global debug flag management
- Conditional output based on debug state

## Integration Points

### Data Input

The system accepts data through:

- Pandas DataFrames
- CSV files
- JSON configurations
- Direct API calls

### Configuration

Components are configured through:

- JSON configuration files
- Command-line arguments
- Environment variables
- Brand-specific config files

### Output Formats

Results can be exported as:

- JSON reports
- CSV files
- Console output
- Custom formats via reporters

## Error Handling

All components implement consistent error handling:

- `ValidationError` for validation failures
- `AnomalyError` for anomaly detections
- Custom exceptions for system errors
- Graceful degradation for missing dependencies

## Performance Considerations

The system is optimized for:

- Large-scale batch processing
- Fast validation with minimal latency
- GPU acceleration (when available)
- Memory-efficient batch processing
- Parallel execution

## Extensibility

New components can be added by:

1. Implementing the appropriate interface
2. Registering with the component registry
3. Adding configuration entries
4. Providing documentation

The modular design ensures that new validators, detectors, and reporters can be added without modifying core system logic.