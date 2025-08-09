# Anomaly Detectors

This module contains three different anomaly detection approaches, each with its own strengths and use cases.

## Detection Methods

### 1. Pattern-Based Detection (`/pattern_based/`)
- **Purpose**: Rule-based anomaly detection using configurable patterns
- **Confidence Level**: Medium (0.7-0.8)
- **No Training Required**: Works with JSON rule files

#### Adding New Fields
1. Create a rule file: `pattern_based/rules/{field_name}.json`
2. Define patterns, known values, and validation rules
3. The detector automatically loads the rules

Example rule file structure:
```json
{
  "field_name": "material",
  "known_values": ["cotton", "polyester", "wool"],
  "format_patterns": [
    {
      "name": "material_format",
      "pattern": "^[a-zA-Z\\s\\-%]+$",
      "message": "Invalid material format"
    }
  ],
  "validation_rules": [
    {
      "name": "not_empty",
      "type": "not_empty",
      "message": "Material cannot be empty"
    }
  ]
}
```

### 2. ML-Based Detection (`/ml_based/`)
- **Purpose**: Semantic similarity-based anomaly detection
- **Confidence Level**: Configurable (default 0.7)
- **Training Required**: Yes, using sentence transformers

#### Key Components
- `ml_anomaly_detector.py`: Main detector implementation
- `model_training.py`: Training script for new models
- `check_anomalies.py`: Anomaly checking logic
- `gpu_utils.py`: GPU acceleration utilities

#### Training Process
1. Prepare clean data samples
2. Configure field in `model_training.py`
3. Run: `python ../main.py ml-train your_data.csv --fields {field_name}`
4. Model saved to `ml_based/models/trained/{field_name}/`

### 3. LLM-Based Detection (`/llm_based/`)
- **Purpose**: Advanced semantic understanding with context
- **Confidence Level**: Configurable (default 0.6)
- **Training Required**: Yes, fine-tuning transformer models

#### Features
- Dynamic context encoding
- Few-shot learning capability
- Temporal awareness
- Prototype-based reprogramming

## Common Interfaces

All detectors implement `AnomalyDetectorInterface`:
- `learn_patterns()`: Initialize/train the detector
- `detect_anomaly()`: Detect anomalies in single values
- `bulk_detect()`: Batch anomaly detection

## Error Injection

The module includes utilities for testing:
- `anomaly_injection.py`: Inject synthetic anomalies
- `anomaly_injection_rules/`: Field-specific injection rules

## Configuration

Each detection method can be configured through:
- Threshold values (confidence levels)
- Model parameters
- Rule definitions
- GPU usage settings

## Model Artifacts
- ML models: `data/models/ml/trained/{field}/{variation}/`
- LLM models: `data/models/llm/{field}_model/{variation}/`