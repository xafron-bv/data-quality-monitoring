# ML-Based Anomaly Detection System Documentation

## Overview

Machine learning approach for anomaly detection in data quality monitoring using **SentenceTransformers** with **triplet loss** training to learn semantic representations that distinguish between normal and anomalous data patterns.

## System Architecture

### Core Components

```
anomaly_detectors/ml_based/
├── ml_anomaly_detector.py          # Main detector implementation
├── model_training.py               # Training logic and evaluation
├── hyperparameter_search.py        # HP optimization
├── check_anomalies.py             # Inference and detection
├── rule_column_map.py             # Column mapping configuration
├── optimal_params.json            # Best parameters per column
└── requirements.txt               # Dependencies
```

### Design Philosophy

**Recall-focused approach**: Prioritizes detection of actual anomalies over minimizing false positives. The system operates under the constraint that **clean reference data is never available during production anomaly detection**.

## Training Architecture

### Triplet Loss Training

The system uses triplet loss to train SentenceTransformer models:

- **Anchor**: Clean, valid data example
- **Positive**: Another clean, valid data example
- **Negative**: Error-injected version (synthetic anomaly)

### Error Injection Strategy

Sophisticated error injection creates realistic anomalies:

- **Unicode errors**: Character encoding corruption
- **Composition errors**: Mathematical inconsistencies
- **Format corruption**: Structure violations
- **Character errors**: Typos, missing symbols
- **Spacing issues**: Whitespace anomalies

### Triplet Dataset Creation

```python
def create_triplet_dataset(clean_texts, error_rules):
    """Creates balanced triplet datasets where:
    - Clean values are similar to each other
    - Anomalies are distant from clean values
    """
    triplets = []
    for anchor in clean_texts:
        for positive in clean_texts:
            if positive != anchor:
                negative = apply_error_rule(anchor, random_rule)
                triplets.append([anchor, positive, negative])
    return triplets
```

## Hyperparameter Optimization

### Search Strategy

**Recall-focused optimization** with precision constraints:
- Primary objective: Maximize recall
- Constraint: Maintain minimum 30% precision
- Method: Random search across parameter space

### Search Space

```json
{
  "model_name": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
  "triplet_margin": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
  "distance_metric": ["COSINE", "EUCLIDEAN"],
  "batch_size": [8, 16, 32, 48, 64],
  "epochs": [2, 3, 4, 5, 6],
  "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5]
}
```

## Anomaly Detection Algorithm

### Production Detection (No Clean References)

```python
def detect_anomalies(model, values, threshold=0.6):
    """Detects anomalies using statistical analysis of embeddings"""
    
    # Get embeddings
    embeddings = model.encode(values)
    
    # Global outlier detection
    centroid = np.mean(embeddings, axis=0)
    centroid_sims = cosine_similarity(embeddings, centroid.reshape(1, -1))
    
    # Local outlier detection
    pairwise_sims = cosine_similarity(embeddings)
    avg_sims = np.mean(pairwise_sims, axis=1)
    
    # Conservative decision (minimum of both)
    final_scores = np.minimum(centroid_sims.flatten(), avg_sims)
    
    return final_scores < threshold
```

### Detection Logic

1. **Global Detection**: Compare to dataset centroid
2. **Local Detection**: Compare to neighboring values
3. **Conservative Decision**: Flag if unusual globally OR locally

## Implementation Details

### Column-Specific Configuration

Each column has optimized settings:

```python
COLUMN_CONFIGS = {
    'material': {
        'model': 'multi-qa-MiniLM-L6-cos-v1',
        'margin': 1.0,
        'epochs': 3
    },
    'color_name': {
        'model': 'multi-qa-MiniLM-L6-cos-v1',
        'margin': 0.8,
        'epochs': 4
    }
}
```

### Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512})
  (1): Pooling({'dimension': 384, 'pooling_mode_mean_tokens': True})
  (2): Normalize()
)
```

## Usage

### Training Models

```bash
# Train with hyperparameter search
python anomaly_detectors/ml_based/model_training.py material data.csv --use-hp-search

# Train with optimal parameters
python anomaly_detectors/ml_based/model_training.py material data.csv
```

### Running Detection

```bash
# Command line
python anomaly_detectors/ml_based/check_anomalies.py data.csv --field material --threshold 0.6

# Programmatic
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector

detector = MLAnomalyDetector("material", threshold=0.6)
detector.learn_patterns(df, "material")
anomalies = detector.bulk_detect(df, "material")
```

## Performance Characteristics

### Model Performance

| Field | Model | Recall | Precision | F1-Score |
|-------|-------|--------|-----------|----------|
| material | multi-qa-MiniLM | 0.88 | 0.65 | 0.75 |
| color_name | multi-qa-MiniLM | 1.00 | 0.85 | 0.92 |
| category | multi-qa-MiniLM | 1.00 | 0.92 | 0.96 |

### Threshold Impact

- **Low threshold (0.1-0.3)**: High precision, may miss subtle anomalies
- **Default (0.6)**: Balanced detection
- **High threshold (0.9+)**: High recall, more false positives

## Key Strengths

- **No reference data required**: Works without clean examples in production
- **Semantic understanding**: Captures meaning beyond surface patterns
- **Column-specific optimization**: Tailored for each data type
- **GPU acceleration**: Efficient processing with hardware support
- **Extensible design**: Easy to add new fields and models

## Integration

The ML detector integrates with the unified detection system:
- Implements standard `AnomalyDetectorInterface`
- Compatible with weighted combination
- Automatic model loading and caching
- Graceful degradation when models unavailable
