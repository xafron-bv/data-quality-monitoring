# Class Hierarchy Documentation

## See Also

- [Main Project README](./README.md)
- [ML Anomaly Detection Documentation](./ML_Anomaly_Detection_Documentation.md)
- [ML Anomaly Detection Overview](./ML_Anomaly_Detection_Overview.md)
- [Pattern-Based Anomaly Detection](./PATTERN_BASED_README.md)

# Data Quality Monitoring System - Class Hierarchy and Interface Documentation

## Table of Contents

- [Overview](#overview)
- [Core Architecture Principles](#core-architecture-principles)
- [Interface Hierarchy](#interface-hierarchy)
  - [1. Detection Interfaces](#1-detection-interfaces)
  - [2. Reporting Interfaces](#2-reporting-interfaces)
- [Error and Result Classes](#error-and-result-classes)
- [Core Implementation Classes](#core-implementation-classes)
- [Evaluation Flow](#evaluation-flow)
- [Detection Method Characteristics](#detection-method-characteristics)
- [Report Format Standardization](#report-format-standardization)
- [Usage Patterns](#usage-patterns)
- [Key Design Benefits](#key-design-benefits)
- [Summary](#summary)

## Overview

This data quality monitoring system implements a multi-layered approach combining rule-based validation, pattern-based anomaly detection, and ML-based anomaly detection through a unified interface architecture.

## Core Architecture Principles

1. **Interface Segregation**: Separate interfaces for detection and reporting
2. **Unified Access**: Common interface for all detection methods via `UnifiedDetectorInterface`
3. **Polymorphic Design**: All detectors implement common base interfaces
4. **Composition over Inheritance**: `Evaluator` composes multiple detector types
5. **Error Standardization**: Common error/result formats across detection methods

## Interface Hierarchy

### 1. Detection Interfaces

#### ValidatorInterface (validators module)
```python
# Abstract interface for rule-based validation
class ValidatorInterface(ABC):
    @abstractmethod
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """Validate data and return validation errors"""
        pass
```

**Purpose**: Rule-based validation with high confidence detection
**Returns**: `List[ValidationError]`
**Implementations**: 
- `MaterialValidator` (validators/material/validate.py)
- `ColorNameValidator` (validators/color_name/validate.py)
- Other field-specific validators

#### AnomalyDetectorInterface (anomaly_detectors/anomaly_detector_interface.py)
```python
class AnomalyDetectorInterface(ABC):
    @abstractmethod
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """Detect anomaly in single value"""
        pass
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """Learn normal patterns from data (optional)"""
        pass
    
    def bulk_detect(self, df: pd.DataFrame, column_name: str, 
                   batch_size: int = None, max_workers: int = 7) -> List[AnomalyError]:
        """Detect anomalies in bulk with parallel processing"""
        pass
```

**Purpose**: Pattern-based and ML-based anomaly detection
**Returns**: `List[AnomalyError]`
**Key Features**:
- Parallel batch processing
- GPU optimization for ML detectors
- Pattern learning capability

**Implementations**:
- Pattern-based detectors (anomaly_detectors/pattern_based/*/detect.py)
- `MLAnomalyDetector` (anomaly_detectors/ml_based/ml_anomaly_detector.py)

#### UnifiedDetectorInterface (unified_detection_interface.py)
```python
class UnifiedDetectorInterface(ABC):
    @abstractmethod
    def detect_issues(self, df: pd.DataFrame, field_name: str,
                     enable_validation: bool = True,
                     enable_anomaly_detection: bool = True,
                     enable_ml_detection: bool = True,
                     validation_threshold: float = 0.0,
                     anomaly_threshold: float = 0.7,
                     ml_threshold: float = 0.7) -> List[UnifiedDetectionResult]:
        """Unified detection using multiple approaches"""
        pass
```

**Purpose**: Combine all detection methods under single interface
**Returns**: `List[UnifiedDetectionResult]`
**Implementation**: `CombinedDetector`

### 2. Reporting Interfaces

#### ReporterInterface (validators module)
```python
class ReporterInterface(ABC):
    @abstractmethod
    def generate_report(self, validation_errors: List[ValidationError], 
                       original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate human-readable reports from validation errors"""
        pass
```

#### AnomalyReporterInterface (anomaly_detectors/reporter_interface.py)
```python
class AnomalyReporterInterface(ABC):
    @abstractmethod
    def generate_report(self, 
                       anomaly_results: Union[List[AnomalyError], List[MLAnomalyResult]], 
                       original_df: pd.DataFrame,
                       threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generate reports from anomaly detection results"""
        pass
```

**Key Features**:
- Supports both `AnomalyError` and `MLAnomalyResult` inputs
- Threshold-based filtering
- Standardized report format

## Error and Result Classes

### ValidationError (validators module)
```python
class ValidationError:
    def __init__(self, error_type: str, probability: float, details: Dict[str, Any]):
        self.error_type = error_type
        self.probability = probability  # 0.0 to 1.0
        self.details = details
        # Context added by bulk_validate:
        self.row_index = None
        self.column_name = None
        self.error_data = None
```

**Purpose**: Represents rule-based validation errors
**Confidence**: High (typically 0.8-1.0)

### AnomalyError (anomaly_detectors/anomaly_error.py)
```python
class AnomalyError:
    def __init__(self, 
                anomaly_type: Union[str, Enum], 
                probability: float, 
                details: Optional[Dict[str, Any]] = None,
                # ML-specific fields:
                feature_contributions: Optional[Dict[str, float]] = None,
                nearest_neighbors: Optional[List[Tuple[int, float]]] = None,
                cluster_info: Optional[Dict[str, Any]] = None,
                probability_info: Optional[Dict[str, Any]] = None,
                explanation: Optional[str] = None):
```

**Purpose**: Represents anomaly detection results (both pattern-based and ML-based)
**Confidence**: Variable (0.0-1.0)
**Key Features**:
- Supports both pattern-based and ML metadata
- Feature contributions for explainability
- Nearest neighbors for context
- Natural language explanations

### MLAnomalyResult (anomaly_detectors/reporter_interface.py)
```python
class MLAnomalyResult:
    def __init__(self, 
                row_index: int, 
                column_name: str,
                value: Any,
                probabiliy: float,  # Note: typo in original
                feature_contributions: Dict[str, float] = None,
                nearest_neighbors: List[Tuple[int, float]] = None,
                cluster_info: Dict[str, Any] = None,
                probability_info: Dict[str, Any] = None,
                explanation: str = None):
```

**Purpose**: Detailed ML-specific anomaly detection results
**Used by**: ML-based reporters for enhanced reporting

### UnifiedDetectionResult (unified_detection_interface.py)
```python
class UnifiedDetectionResult:
    def __init__(self,
                 row_index: int,
                 field_name: str,
                 value: Any,
                 detection_type: DetectionType,  # VALIDATION, ANOMALY, ML_ANOMALY
                 probability: float,
                 error_code: str,
                 message: str,
                 details: Dict[str, Any] = None,
                 ml_features: Dict[str, Any] = None):
```

**Purpose**: Unified representation of all detection types
**Key Features**:
- Converts from `ValidationError`, `AnomalyError`, `MLAnomalyResult`
- Consistent interface across detection types
- Detection type tracking

## Core Implementation Classes

### 1. Evaluator (evaluator.py)
**Primary orchestration class that coordinates all detection and evaluation**

```python
class Evaluator:
    def __init__(self, 
                 validator: Optional[ValidatorInterface] = None,
                 validator_reporter: Optional[ReporterInterface] = None,
                 anomaly_detector: Optional[AnomalyDetectorInterface] = None,
                 anomaly_reporter: Optional[AnomalyReporterInterface] = None,
                 ml_detector: Optional[MLAnomalyDetector] = None,
                 ml_reporter: Optional[MLAnomalyReporter] = None):
```

**Key Methods**:
- `evaluate_unified()`: Uses unified detection interface
- `evaluate_field()`: Individual approach evaluation
- `evaluate_sample()`: Complete sample evaluation with metrics
- `_calculate_metrics()`: Performance metrics calculation

**Composition Pattern**: Aggregates all detector and reporter types

### 2. CombinedDetector (unified_detection_interface.py)
**Unified detector implementation**

```python
class CombinedDetector(UnifiedDetectorInterface):
    def detect_issues(self, df: pd.DataFrame, field_name: str, ...) -> List[UnifiedDetectionResult]:
        # 1. Run validation if enabled
        # 2. Run pattern-based anomaly detection if enabled  
        # 3. Run ML-based detection if enabled
        # 4. Convert all results to UnifiedDetectionResult
        # 5. Apply thresholds and return unified results
```

**Key Features**:
- Threshold-based filtering per detection type
- Result conversion and standardization
- Field name to column name mapping

### 3. MLAnomalyDetector (anomaly_detectors/ml_based/ml_anomaly_detector.py)
**ML-based anomaly detector using sentence transformers**

```python
class MLAnomalyDetector(AnomalyDetectorInterface):
    def __init__(self, field_name: str, results_dir: str = None, 
                 threshold: float = 0.6, use_gpu: bool = True):
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """Load pre-trained model for the field"""
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """Detect anomaly using centroid distance method"""
```

**Key Features**:
- Pre-trained sentence transformer models
- GPU acceleration support
- Centroid-based anomaly detection
- Model caching for performance
- Factory pattern for creation

### 4. ErrorInjector (error_injection.py)
**Error injection for evaluation and testing**

```python
class ErrorInjector:
    def __init__(self, rules: List[Dict[str, Any]]):
    
    def inject_errors(self, df: pd.DataFrame, field_name: str, 
                     max_errors: int = 3, error_probability: float = 0.1) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Inject errors based on rules for testing"""
```

**Purpose**: Generate corrupted data for performance evaluation
**Used by**: Evaluation system to test detector recall/precision

## Evaluation Flow

### 1. Individual Approach Evaluation
```python
# In evaluator.py - evaluate_field()
validation_errors = validator.bulk_validate(df, column_name)
validation_report = validator_reporter.generate_report(validation_errors, df)

anomalies = anomaly_detector.bulk_detect(df, column_name)
anomaly_report = anomaly_reporter.generate_report(anomalies, df)
```

### 2. Unified Approach Evaluation
```python
# In evaluator.py - evaluate_unified()
unified_results = combined_detector.detect_issues(df, field_name, ...)
unified_reports = unified_reporter.generate_report(unified_results, df)
```

### 3. Performance Metrics Calculation
```python
# In evaluator.py - _calculate_metrics()
detected_row_indices = set(result["row_index"] for result in validation_results)
error_row_indices = set(error["row_index"] for error in injected_errors)

true_positives = detected_row_indices.intersection(error_row_indices)
false_positives = detected_row_indices - error_row_indices
false_negatives = error_row_indices - detected_row_indices

precision = len(true_positives) / len(detected_row_indices) if detected_row_indices else 1.0
recall = len(true_positives) / len(error_row_indices) if error_row_indices else 1.0
f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
```

## Detection Method Characteristics

### 1. Rule-Based Validation
- **Interface**: `ValidatorInterface`
- **Confidence**: High (0.8-1.0)
- **Purpose**: Business rule violations
- **Examples**: Invalid formats, missing required fields
- **Performance**: Fast, deterministic

### 2. Pattern-Based Anomaly Detection  
- **Interface**: `AnomalyDetectorInterface`
- **Confidence**: Medium (0.5-0.9)
- **Purpose**: Statistical outliers, unusual patterns
- **Examples**: Unusual text lengths, character patterns
- **Performance**: Medium, pattern-based logic

### 3. ML-Based Anomaly Detection
- **Interface**: `AnomalyDetectorInterface` + `MLAnomalyDetector`
- **Confidence**: Variable (0.0-1.0)
- **Purpose**: Semantic anomalies, context-aware detection
- **Examples**: Semantically incorrect values, unusual combinations
- **Performance**: GPU-accelerated, model-dependent

## Report Format Standardization

All reporters produce standardized output:

```python
{
    'row_index': int,
    'field_name': str,        # Unified: field name
    'column_name': str,       # Individual: column name  
    'value': Any,
    'detection_type': str,    # 'validation', 'anomaly', 'ml_anomaly'
    'probability': float,
    'error_code': str,
    'display_message': str,
    'details': Dict[str, Any],
    'ml_features': Dict[str, Any]  # ML-specific features
}
```

## Usage Patterns

### 1. Individual Detection
```python
# Create specific detectors
validator = MaterialValidator()
anomaly_detector = MaterialAnomalyDetector()
ml_detector = MLAnomalyDetector("material")

# Run detection
validation_errors = validator.bulk_validate(df, "material")
anomalies = anomaly_detector.bulk_detect(df, "material") 
ml_anomalies = ml_detector.bulk_detect(df, "material")
```

### 2. Unified Detection
```python
# Create combined detector
combined_detector = CombinedDetector(
    validator=validator,
    anomaly_detector=anomaly_detector,
    ml_detector=ml_detector
)

# Run unified detection
results = combined_detector.detect_issues(df, "material")
```

### 3. Complete Evaluation
```python
# Create evaluator with all components
evaluator = Evaluator(
    validator=validator,
    validator_reporter=reporter,
    anomaly_detector=anomaly_detector,
    anomaly_reporter=anomaly_reporter,
    ml_detector=ml_detector
)

# Run comprehensive evaluation
results = evaluator.evaluate_sample(df, "material", injected_errors)
```

## Key Design Benefits

1. **Modularity**: Each detection method is independently implementable
2. **Consistency**: Unified interfaces and result formats
3. **Flexibility**: Mix and match detection methods as needed
4. **Performance**: GPU acceleration and parallel processing
5. **Extensibility**: Easy to add new detection methods
6. **Testability**: Error injection framework for performance evaluation
7. **Observability**: Comprehensive reporting and metrics

This architecture enables the system to provide comprehensive data quality monitoring while maintaining clean separation of concerns and consistent interfaces across all detection approaches.

## Summary

### Class Hierarchy Overview

The data quality monitoring system implements a sophisticated multi-layered architecture with the following key characteristics:

#### 1. **Three-Tier Detection Architecture**
- **Validation Layer**: High-confidence rule-based detection (`ValidatorInterface`)
- **Pattern Anomaly Layer**: Medium-confidence pattern-based detection (`AnomalyDetectorInterface`)  
- **ML Anomaly Layer**: Variable-confidence semantic detection (`MLAnomalyDetector`)

#### 2. **Unified Interface Design**
- **Individual Access**: Direct use of specific detector interfaces
- **Unified Access**: Combined detection through `UnifiedDetectorInterface`
- **Evaluation Orchestration**: Complete evaluation through `Evaluator` class

#### 3. **Standardized Result Formats**
- **ValidationError**: Rule-based validation results
- **AnomalyError**: Anomaly detection results (supports ML features)
- **MLAnomalyResult**: ML-specific detailed results
- **UnifiedDetectionResult**: Unified representation across all types

#### 4. **Performance-Oriented Implementation**
- **Parallel Processing**: Batch processing with configurable workers
- **GPU Acceleration**: CUDA/MPS support for ML detection
- **Model Caching**: Shared model instances across detector instances
- **Threshold Filtering**: Configurable confidence thresholds per detection type

#### 5. **Comprehensive Evaluation Framework**
- **Error Injection**: Systematic corruption for testing (`ErrorInjector`)
- **Performance Metrics**: Precision, recall, F1 score calculation
- **Multi-Approach Testing**: Individual and unified approach evaluation
- **Detailed Reporting**: Human-readable and JSON output formats

### Evaluation Usage Patterns

#### Simple Field Analysis
```bash
python evaluate.py data.csv --field material --run validation
```

#### Multi-Method Analysis  
```bash
python evaluate.py data.csv --field material --ml-detector --run all
```

#### Batch Evaluation
```bash
./run_evaluations.sh  # Evaluates all configured fields
```

### Key Integration Points

1. **Field-Column Mapping**: `field_column_map.py` translates field names to CSV column names
2. **Error Rules**: `error_injection_rules/*.json` define field-specific corruption patterns
3. **ML Models**: `anomaly_detectors/ml_based/results/` contains trained models
4. **Configuration**: `optimal_params.json` stores hyperparameter-optimized settings

### Extensibility

The system is designed for easy extension:

- **New Validators**: Implement `ValidatorInterface` + `ReporterInterface`
- **New Anomaly Detectors**: Implement `AnomalyDetectorInterface` + `AnomalyReporterInterface`
- **New ML Models**: Extend `MLAnomalyDetector` or create new implementations
- **New Fields**: Add field mappings and error injection rules

This architecture provides a robust, scalable foundation for comprehensive data quality monitoring across diverse data types and quality requirements. 