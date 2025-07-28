# Data Quality Detection System - Documentation Index

## Overview

This documentation provides comprehensive technical information about the Data Quality Detection System, a multi-layered framework for identifying data quality issues in product datasets.

## Documentation Structure

### 1. [Architecture Overview](ARCHITECTURE.md)
- System design and component relationships
- Data flow and processing pipeline
- Dependency graph and module interactions
- Configuration and extensibility

### 2. [Validation Methods](VALIDATION_METHODS.md)
- Rule-based validation framework
- Field-specific validators
- Error codes and confidence scoring
- Extending validation rules

### 3. [Pattern-Based Detection](PATTERN_DETECTION.md)
- Statistical anomaly detection
- Pattern matching and format validation
- Domain-specific rules
- Configuration and customization

### 4. [ML-Based Detection](ML_DETECTION.md)
- Embedding-based anomaly detection
- Sentence Transformer architecture
- Model training and fine-tuning
- GPU acceleration and optimization

### 5. [LLM-Based Detection](LLM_DETECTION.md)
- Transformer-based anomaly detection
- Context-aware detection strategies
- Few-shot learning capabilities
- Advanced features and optimization

### 6. [Weighted Combination](WEIGHTED_COMBINATION.md)
- Ensemble detection methodology
- Weight optimization strategies
- Performance metrics and evaluation
- Integration best practices

### 7. [Class Hierarchy](CLASS_HIERARCHY_DOCUMENTATION.md)
- Complete class structure
- Interface definitions
- Implementation details
- Code examples

## Quick Reference

### Key Interfaces

- `ValidatorInterface`: Base class for all validators
- `AnomalyDetectorInterface`: Base class for anomaly detectors
- `DetectionResult`: Unified result format
- `FieldMapper`: Field name resolution

### Main Entry Points

- `single_sample_multi_field_demo.py`: Single sample analysis
- `multi_sample_evaluation.py`: Statistical evaluation
- `comprehensive_detector.py`: Detection orchestration
- `evaluator.py`: Evaluation framework

### Configuration Files

- `brand_configs/*.json`: Brand-specific configurations
- `detection_weights.json`: Optimized detection weights
- `validators/error_injection_rules/*.json`: Error injection rules
- `anomaly_detectors/pattern_based/rules/*.json`: Pattern rules

## Getting Started

1. Start with [Architecture Overview](ARCHITECTURE.md) for system understanding
2. Review detection method docs based on your needs
3. Check [Class Hierarchy](CLASS_HIERARCHY_DOCUMENTATION.md) for implementation details
4. Refer to main [README](../README.md) for setup and usage

## Performance Guidelines

- Use GPU acceleration for ML/LLM methods when available
- Batch processing for large datasets
- Enable only required detection methods
- Tune thresholds based on your use case

## Development Guidelines

- Follow existing interfaces when adding new detectors
- Document error codes and confidence scores
- Include error injection rules for testing
- Update relevant documentation

## Support

For questions or issues:
1. Check existing documentation
2. Review code comments and docstrings
3. Examine test cases for usage examples