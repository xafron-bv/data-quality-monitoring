# Data Quality Detection System - Documentation Index

## Overview

This documentation provides comprehensive technical information about the Data Quality Detection System, a multi-layered framework for identifying data quality issues in product datasets.

## Documentation Structure

### High-Level Documentation (docs/)

1. **[Architecture Overview](ARCHITECTURE.md)**
   - System design and component relationships
   - Data flow and processing pipeline
   - Dependency graph and module interactions
   - Configuration and extensibility

2. **[Weighted Combination](WEIGHTED_COMBINATION.md)**
   - Ensemble detection methodology
   - Weight optimization strategies
   - Performance metrics and evaluation

3. **[Class Hierarchy](CLASS_HIERARCHY_DOCUMENTATION.md)**
   - Complete class structure
   - Interface definitions
   - Implementation details

### Component-Specific Documentation

4. **[Validation Methods](../validators/README.md)**
   - Rule-based validation framework
   - Field-specific validators
   - Error codes and confidence scoring
   - Location: `validators/README.md`

5. **[Pattern-Based Detection](../anomaly_detectors/pattern_based/README.md)**
   - Statistical anomaly detection
   - Pattern matching and format validation
   - Domain-specific rules
   - Location: `anomaly_detectors/pattern_based/README.md`

6. **[ML-Based Detection](../anomaly_detectors/ml_based/README.md)**
   - Embedding-based anomaly detection
   - Sentence Transformer architecture
   - Model training and fine-tuning
   - Location: `anomaly_detectors/ml_based/README.md`

7. **[LLM-Based Detection](../anomaly_detectors/llm_based/TECHNICAL_README.md)**
   - Transformer-based anomaly detection
   - Context-aware detection strategies
   - Few-shot learning capabilities
   - Location: `anomaly_detectors/llm_based/TECHNICAL_README.md`

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

## Documentation Philosophy

- **High-level docs** in `docs/`: Architecture, system design, cross-cutting concerns
- **Component docs** in their directories: Implementation details, usage, configuration
- **Technical depth**: Assumes familiarity with ML/NLP concepts
- **No redundancy**: Avoids documenting what's easily found in code

## Getting Started

1. Start with [Architecture Overview](ARCHITECTURE.md) for system understanding
2. Review component-specific docs based on your area of interest
3. Check implementation details in the relevant component directories
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
- Place documentation in the appropriate directory:
  - System-wide concepts → `docs/`
  - Component-specific → component directory

## Support

For questions or issues:
1. Check relevant documentation in component directories
2. Review code comments and docstrings
3. Examine test cases for usage examples