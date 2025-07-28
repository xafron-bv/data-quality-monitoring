# Detection System Analysis Report

## 1. Entry Point Analysis

### Primary Entry Points

#### a) **single_sample_multi_field_demo.py**
- **Purpose**: Main demonstration script showcasing comprehensive data quality monitoring capabilities
- **Functionality**: 
  - Generates synthetic samples with configurable error injection
  - Runs all detection methods (validation, pattern-based, ML-based, LLM-based)
  - Produces consolidated reports compatible with HTML5 visualization
- **Execution Context**: Development/demonstration environment for showcasing system capabilities
- **Rationale**: Provides a single-sample comprehensive approach for efficient testing and demonstration

#### b) **multi_sample_evaluation.py**
- **Purpose**: Batch evaluation of detection methods across multiple samples
- **Functionality**:
  - Processes multiple data samples with various error patterns
  - Evaluates detection performance with metrics (precision, recall, F1)
  - Supports cross-validation and performance comparison
- **Execution Context**: Performance evaluation and model tuning
- **Rationale**: Enables systematic evaluation of detection accuracy across diverse scenarios

#### c) **comprehensive_detector.py**
- **Purpose**: Core detection orchestrator (can be run standalone)
- **Functionality**:
  - Coordinates all detection methods for field-by-field analysis
  - Implements both priority-based and weighted combination approaches
  - Manages memory efficiently through sequential processing
- **Execution Context**: Production detection operations
- **Rationale**: Central hub for detection logic, reusable across different contexts

#### d) **ml_curve_generator.py**
- **Purpose**: ML model performance analysis and visualization
- **Functionality**:
  - Generates performance curves for ML detection models
  - Analyzes threshold sensitivity
  - Produces visualization plots
- **Execution Context**: Model optimization and threshold tuning
- **Rationale**: Essential for understanding ML model behavior and setting optimal thresholds

#### e) **generate_detection_weights.py**
- **Purpose**: Generates field-specific weights for weighted combination detection
- **Functionality**:
  - Analyzes historical performance data
  - Calculates optimal weights based on F1 scores
  - Produces weights configuration file
- **Execution Context**: Post-evaluation optimization
- **Rationale**: Enables adaptive weighting based on actual performance data

#### f) **detection_comparison.py**
- **Purpose**: Compares different detection approaches
- **Functionality**:
  - Runs side-by-side comparison of detection methods
  - Generates comparative reports
- **Execution Context**: Method selection and optimization
- **Rationale**: Helps choose the best detection strategy for specific use cases

### Utility Entry Points

#### g) **analyze_column.py**
- **Purpose**: Column-level data analysis tool
- **Functionality**: Analyzes data distribution and patterns in specific columns
- **Execution Context**: Data exploration and understanding
- **Rationale**: Helps understand data characteristics before detection

#### h) **error_injection.py**
- **Purpose**: Error injection utility for testing
- **Functionality**: Applies configurable error patterns to clean data
- **Execution Context**: Test data generation
- **Rationale**: Creates realistic test scenarios for validation

#### i) **anomaly_detectors/anomaly_injection.py**
- **Purpose**: Anomaly injection for pattern-based testing
- **Functionality**: Injects pattern-based anomalies into data
- **Execution Context**: Test data generation for anomaly detection
- **Rationale**: Separate from error injection to handle pattern-specific anomalies

## 2. Architecture Assessment

### Overall Architecture

The system follows a **modular, plugin-based architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points Layer                        │
│  (demo scripts, evaluation tools, comparison utilities)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│  (ComprehensiveFieldDetector, Evaluator, UnifiedInterface)  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Detection Methods Layer                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Validation    │  Pattern-Based  │    ML/LLM-Based         │
│   (Rule-based)  │   (Anomaly)     │    (Semantic)           │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                       │
│  (FieldMapper, BrandConfig, ErrorInjection, Reporters)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│          (CSV files, JSON configs, Model files)              │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

1. **Detection Modules** (`/anomaly_detectors/`, `/validators/`)
   - Each detection method is self-contained
   - Implements common interfaces for consistency
   - Field-specific logic isolated in configuration files

2. **Configuration System**
   - `brand_config.py/json`: Brand-specific field mappings
   - `field_mapper.py`: Translation between standard fields and column names
   - `detection_weights.json`: Performance-based method weights

3. **Reporting Infrastructure**
   - `consolidated_reporter.py`: Unified reporting format
   - `confusion_matrix_analyzer.py`: Performance analysis
   - Field-specific reporters in respective modules

### Design Patterns

1. **Strategy Pattern**: Different detection methods implement common interfaces
2. **Factory Pattern**: Dynamic loading of field-specific detectors
3. **Observer Pattern**: Reporters observe and format detection results
4. **Singleton/Caching**: Model caching to optimize memory usage

### Class Hierarchy

```
AnomalyDetectorInterface (ABC)
├── PatternBasedDetector
├── MLAnomalyDetector
└── LLMAnomalyDetector

ValidatorInterface (ABC)
└── Field-specific validators (e.g., MaterialValidator)

ReporterInterface (ABC)
├── AnomalyReporter
├── ValidatorReporter
└── UnifiedReporter

FieldMapper
└── Brand-specific mappers (loaded from config)
```

## 3. Detection Methods Documentation

### a) Validation (Rule-Based Detection)
- **Location**: `/validators/{field_name}/validate.py`
- **Purpose**: High-confidence error detection using business rules
- **Error Validation**:
  - Format validation (regex patterns)
  - Business rule enforcement
  - Domain-specific constraints
- **Training**: No training required - rules are manually defined
- **Adding New Fields**:
  1. Create folder `/validators/{field_name}/`
  2. Add `validate.py` with Validator class
  3. Add `error_messages.json` for error descriptions
  4. Follow template structure from existing validators

### b) Pattern-Based Anomaly Detection
- **Location**: `/anomaly_detectors/pattern_based/`
- **Purpose**: Medium-confidence anomaly detection using patterns
- **Error Validation**:
  - Known value validation
  - Format pattern matching
  - Statistical outlier detection
- **Training**: No training - uses rule files
- **Adding New Fields**:
  1. Create `/anomaly_detectors/pattern_based/rules/{field_name}.json`
  2. Define patterns, known values, and validation rules
  3. System automatically loads rules for new fields

### c) ML-Based Anomaly Detection
- **Location**: `/anomaly_detectors/ml_based/`
- **Purpose**: Semantic similarity-based anomaly detection
- **Error Validation**:
  - Sentence transformer embeddings
  - Cosine similarity to reference centroid
  - Threshold-based classification
- **Training Process**:
  1. Run `model_training.py` with field data
  2. Creates triplet dataset (clean, similar, anomalous)
  3. Fine-tunes sentence transformer
  4. Generates reference centroid from clean data
- **Adding New Fields**:
  1. Add field configuration in `model_training.py`
  2. Prepare training data
  3. Run training script
  4. Model saved automatically in results folder

### d) LLM-Based Anomaly Detection
- **Location**: `/anomaly_detectors/llm_based/`
- **Purpose**: Advanced semantic understanding using language models
- **Error Validation**:
  - Dynamic context encoding
  - Few-shot learning capability
  - Prototype-based reprogramming
- **Training Process**:
  1. Run `llm_model_training.py`
  2. Fine-tunes transformer model for classification
  3. Supports temporal and contextual features
- **Adding New Fields**:
  1. Prepare labeled training data
  2. Configure model parameters
  3. Run training with appropriate context columns

## 4. Brand Support Expansion

### Current Implementation
- Single-brand focus with configurable field mappings
- Brand configuration in `brand_config.json`

### Required Modifications for Multi-Brand Support

1. **Configuration Structure**:
   ```json
   {
     "brands": {
       "brand_a": {
         "field_mappings": {...},
         "default_data_path": "...",
         "custom_thresholds": {...}
       },
       "brand_b": {...}
     }
   }
   ```

2. **Code Changes**:
   - Enhance `BrandConfig` class to support brand selection
   - Update `FieldMapper` to handle brand-specific mappings
   - Modify detection methods to use brand-specific models/rules

3. **Model/Rule Organization**:
   ```
   /models/
     /brand_a/
       /ml_models/
       /rules/
     /brand_b/
       /ml_models/
       /rules/
   ```

4. **Scalable Approach**:
   - Implement brand registry system
   - Create brand-specific model namespaces
   - Add brand parameter to all entry points
   - Implement cross-brand performance comparison

## 5. Evaluation Framework

### HTML5 Viewer Integration
- **Purpose**: Visual inspection of detection results
- **Usage**:
  1. Generate detection reports using demo scripts
  2. Open `data_quality_viewer.html` in browser
  3. Upload CSV data file
  4. Upload JSON detection report
  5. Interactively explore detected issues

### Available Metrics

1. **Detection Metrics**:
   - **Precision**: Accuracy of positive detections
   - **Recall**: Coverage of actual errors
   - **F1 Score**: Harmonic mean of precision and recall
   - **Accuracy**: Overall correctness

2. **Confusion Matrix Components**:
   - True Positives (TP): Correctly detected issues
   - False Positives (FP): Incorrectly flagged as issues
   - False Negatives (FN): Missed real issues
   - True Negatives (TN): Correctly identified clean data

3. **Field-Level Metrics**:
   - Per-field precision/recall/F1
   - Detection method performance comparison
   - Issue type distribution

4. **Aggregate Metrics**:
   - Overall system performance
   - Method-wise contribution
   - Weighted combination effectiveness

### Improving Detection Accuracy

1. **Threshold Tuning**:
   - Use `ml_curve_generator.py` to analyze threshold impact
   - Adjust field-specific thresholds based on precision-recall trade-off

2. **Weight Optimization**:
   - Run `generate_detection_weights.py` after evaluation
   - Use weighted combination mode for better accuracy

3. **Model Retraining**:
   - Collect false positives/negatives
   - Augment training data
   - Retrain ML/LLM models with improved data

4. **Rule Refinement**:
   - Analyze false detections
   - Update validation rules and patterns
   - Add domain-specific knowledge

5. **Feature Engineering**:
   - Add contextual features for LLM detection
   - Implement temporal awareness
   - Use cross-field validation

## Key Insights

1. **Modular Architecture**: Each detection method is independent and pluggable
2. **Field-Agnostic Design**: New fields can be added without code changes
3. **Performance-Aware**: Weighted combination adapts to actual performance
4. **Memory Efficient**: Sequential processing and model caching
5. **Production Ready**: Clear interfaces and comprehensive error handling