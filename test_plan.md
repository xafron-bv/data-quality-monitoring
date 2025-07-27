# Testing Plan for Import Reorganization

## Overview
This plan outlines the systematic testing of all modified scripts to ensure they work correctly after import reorganization.

## Environment Setup
1. Check Python version and available packages
2. Create test data if needed
3. Set up brand configuration

## Testing Categories

### 1. Core Utility Scripts
These can be tested with minimal setup:
- `field_mapper.py` - Test import and basic functionality
- `common_interfaces.py` - Test import after FieldMapper extraction
- `exceptions.py` - Test import
- `debug_config.py` - Test import

### 2. Analysis and Helper Scripts
These need CSV data but can run independently:
- `analyze_column.py` - Requires CSV file and brand
- `field_column_map.py` - Test import and function calls
- `brand_configs.py` - Test brand management functions
- `manage_brands.py` - Test brand management CLI

### 3. Error and Anomaly Injection Scripts
- `error_injection.py` - Test loading rules and basic injection
- `anomaly_detectors/anomaly_injection.py` - Test loading anomaly rules

### 4. Detection Components
Test individual detection modules:
- `anomaly_detectors/ml_based/ml_anomaly_detector.py` - Test import (needs ML dependencies)
- `anomaly_detectors/llm_based/llm_anomaly_detector.py` - Test import (needs LLM dependencies)
- `anomaly_detectors/pattern_based/pattern_based_detector.py` - Test basic functionality

### 5. Training Scripts
These require more setup and data:
- `anomaly_detectors/ml_based/model_training.py` - Requires training data
- `anomaly_detectors/ml_based/hyperparameter_search.py` - Requires data
- `anomaly_detectors/ml_based/generate_centroids_for_existing_models.py` - Requires trained models

### 6. Evaluation and Demo Scripts
- `evaluator.py` - Test import and basic structure
- `evaluate.py` - Main evaluation script (requires full setup)
- `demo.py` - Main demo script (requires data and brand)
- `comprehensive_detector.py` - Core detection orchestrator
- `comprehensive_sample_generator.py` - Sample generation

### 7. Reporting and Analysis Scripts
- `ml_curve_generator.py` - Requires data and trained models
- `confusion_matrix_analyzer.py` - Requires detection results
- `consolidated_reporter.py` - Test import
- `detection_comparison.py` - Requires results data

### 8. Integration Tests
- `test_llm_integration.py` - Test LLM integration
- `unified_detection_interface.py` - Test unified interface

## Testing Approach
1. Start with simple import tests
2. Create minimal test data
3. Test scripts that don't require trained models
4. Test scripts that can work with mock data
5. Document any scripts that require full environment setup

## Expected Issues
- Missing dependencies (pandas, numpy, etc.)
- Missing trained models
- Missing configuration files
- Missing test data