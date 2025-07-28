# Data Quality Detection System Architecture

## System Overview

The Data Quality Detection System is a comprehensive framework for detecting data quality issues, anomalies, and validation errors in product data. It employs a multi-layered detection approach combining rule-based validation, statistical pattern detection, machine learning, and large language models.

## Core Architecture Principles

### 1. **Unified Detection Interface**
- All detection methods implement common interfaces (`DetectionResult`, `AnomalyDetectorInterface`, `ValidatorInterface`)
- Enables seamless integration and comparison of different detection approaches
- Supports pluggable detection components

### 2. **Field-Aware Processing**
- Field-specific detection logic encapsulated in dedicated modules
- Dynamic field mapping through `FieldMapper` and brand configurations
- Supports multiple brands with different field naming conventions

### 3. **Multi-Method Detection Pipeline**
- **Validation Layer**: Rule-based checks for format, constraints, and business logic
- **Pattern Detection**: Statistical anomaly detection based on field-specific patterns
- **ML Detection**: Embedding-based anomaly detection using Sentence Transformers
- **LLM Detection**: Context-aware detection using fine-tuned language models

## Component Architecture

### Detection Pipeline Flow

```
Raw Data → Field Mapper → Detection Pipeline → Results Aggregation → Reporting
                              ├── Validators
                              ├── Pattern Detector
                              ├── ML Detector
                              └── LLM Detector
```

### Key Components

1. **Field Mapper** (`field_mapper.py`)
   - Maps logical field names to actual column names
   - Handles brand-specific field naming conventions
   - Validates column existence and data types

2. **Comprehensive Detector** (`comprehensive_detector.py`)
   - Orchestrates all detection methods
   - Manages detection configuration and thresholds
   - Aggregates results from multiple detectors
   - Handles error injection for testing

3. **Evaluator** (`evaluator.py`)
   - Coordinates validation and anomaly detection
   - Generates unified detection reports
   - Manages batch processing and parallelization

4. **Detection Methods**:
   - **Validators** (`validators/`): Field-specific validation rules
   - **Pattern-Based** (`anomaly_detectors/pattern_based/`): Rule-based pattern matching
   - **ML-Based** (`anomaly_detectors/ml_based/`): Embedding similarity detection
   - **LLM-Based** (`anomaly_detectors/llm_based/`): Fine-tuned language models

## Data Flow

### 1. **Input Processing**
```python
DataFrame → BrandConfig → FieldMapper → Column Resolution
```

### 2. **Detection Execution**
```python
For each field:
  → Parallel execution of all enabled detectors
  → Each detector returns List[DetectionResult]
  → Results aggregated by ComprehensiveDetector
```

### 3. **Result Aggregation**
```python
DetectionResults → ConfusionMatrixAnalyzer → Performance Metrics
                → ConsolidatedReporter → Human-readable reports
```

## Dependency Graph

### Core Dependencies
- **pandas**: Data manipulation and processing
- **numpy**: Numerical operations
- **scikit-learn**: ML utilities and metrics
- **sentence-transformers**: Embedding generation for ML detection
- **transformers**: LLM model loading and fine-tuning
- **torch**: Deep learning backend

### Module Dependencies
```
common_interfaces.py
    ↓
field_mapper.py ← brand_config.py
    ↓
validators/ ← validation_error.py
anomaly_detectors/ ← anomaly_error.py
    ↓
comprehensive_detector.py
    ↓
evaluator.py
    ↓
multi_sample_evaluation.py / single_sample_multi_field_demo.py
```

## Configuration System

### Brand Configuration
- JSON-based configuration in `brand_configs/`
- Defines field mappings, thresholds, and enabled fields
- Supports multiple brands with different schemas

### Detection Weights
- Optimized weights generated from evaluation results
- Stored in `detection_weights.json`
- Used for weighted combination of detection methods

### Error Injection Rules
- Validation errors: `validators/error_injection_rules/`
- Anomaly patterns: `anomaly_detectors/anomaly_injection_rules/`
- Supports controlled testing and evaluation

## Parallelization Strategy

1. **Batch Processing**: Configurable batch sizes for large datasets
2. **Multi-threading**: Parallel execution of detectors per field
3. **GPU Acceleration**: ML and LLM models support GPU inference
4. **Caching**: Model and embedding caching for performance

## Exception Handling

- Custom exception hierarchy in `exceptions.py`
- Graceful degradation when detectors fail
- Detailed error reporting with context

## Extensibility Points

1. **New Field Types**: Add validator in `validators/` and detection rules
2. **New Detection Methods**: Implement `AnomalyDetectorInterface`
3. **New Brands**: Add configuration in `brand_configs/`
4. **Custom Reporters**: Implement `ReporterInterface`

## Performance Considerations

- **Model Caching**: ML/LLM models cached across runs
- **Batch Processing**: Configurable batch sizes for memory management
- **Selective Detection**: Enable/disable specific detectors as needed
- **Threshold Tuning**: Adjustable confidence thresholds per method