# ML-Based Anomaly Detection

Machine learning approach for semantic anomaly detection using sentence transformers and triplet loss training.

## Architecture Overview

The ML-based detector uses **semantic embeddings** to identify anomalies by learning representations that distinguish normal from anomalous patterns.

### Key Principles
- **Triplet Loss Training**: Learn to separate normal and anomalous examples in embedding space
- **Reference-Free Detection**: Detect anomalies without clean reference data in production
- **Semantic Understanding**: Capture meaning beyond surface-level patterns

## System Components

```
ml_based/
├── ml_anomaly_detector.py      # Main detector class implementing AnomalyDetectorInterface
├── model_training.py           # Training pipeline with triplet loss
├── hyperparameter_search.py    # Automated hyperparameter optimization
├── check_anomalies.py         # Standalone inference script
├── generate_centroids.py      # Centroid generation for trained models
├── optimal_params.json        # Pre-optimized hyperparameters per field
└── results/                   # Trained models directory
```

## Technical Design

### Training Architecture

1. **Data Preparation**
   - Clean data extraction from source
   - Synthetic anomaly generation via error injection
   - Triplet dataset creation (anchor, positive, negative)

2. **Model Training**
   - Base model: Pre-trained sentence transformers
   - Loss function: Triplet loss with configurable margin
   - Optimization: Recall-focused with precision constraints

3. **Hyperparameter Search**
   - Objective: Maximize recall while maintaining 30%+ precision
   - Search space: Model architecture, margin, batch size, learning rate
   - Method: Random search with cross-validation

### Detection Algorithm

```python
# Simplified detection logic
def detect_anomalies(embeddings, threshold):
    # Global anomaly detection
    centroid = embeddings.mean(axis=0)
    global_scores = cosine_similarity(embeddings, centroid)
    
    # Local anomaly detection
    pairwise_sim = cosine_similarity(embeddings)
    local_scores = pairwise_sim.mean(axis=1)
    
    # Conservative combination
    final_scores = minimum(global_scores, local_scores)
    return final_scores < threshold
```

## Configuration

### Field-Specific Models

Each field has optimized model configurations stored in `optimal_params.json`:

```json
{
  "material": {
    "model_name": "multi-qa-MiniLM-L6-cos-v1",
    "triplet_margin": 1.0,
    "distance_metric": "COSINE",
    "batch_size": 48,
    "epochs": 3,
    "learning_rate": 5e-6
  }
}
```

### Thresholds

- **Conservative (0.3-0.5)**: High precision, catch obvious anomalies
- **Balanced (0.6)**: Default, good precision/recall trade-off
- **Aggressive (0.8-0.95)**: High recall, more false positives

## Usage

### Training

```bash
# Train with hyperparameter search
python model_training.py material data.csv --use-hp-search

# Train with pre-optimized parameters
python model_training.py material data.csv
```

### Detection

```python
from ml_anomaly_detector import MLAnomalyDetector

# Initialize detector
detector = MLAnomalyDetector(
    field_name="material",
    threshold=0.6,
    use_gpu=True
)

# Load trained model
detector.learn_patterns(df, "material")

# Detect anomalies
results = detector.bulk_detect(df, "material")
```

## Performance Characteristics

### Strengths
- **Semantic understanding**: Catches conceptually wrong values
- **No reference data needed**: Works with production data alone
- **GPU acceleration**: Fast inference with batch processing
- **Robust to variations**: Handles synonyms and paraphrases

### Limitations
- **Requires training data**: Needs clean examples for training
- **Field-specific models**: Each field needs separate model
- **Memory usage**: Stores embeddings for all values
- **Black box**: Less interpretable than rule-based methods

## Integration

The ML detector integrates seamlessly with the unified detection system:

- Implements `AnomalyDetectorInterface`
- Supports parallel batch processing
- Compatible with weighted combination
- Provides confidence scores and explanations

## Model Management

### Training Pipeline
1. Collect clean training data
2. Generate synthetic anomalies
3. Train with hyperparameter optimization
4. Validate performance metrics
5. Save model and configuration

### Model Storage
- Models saved in `results/results_{field_name}/`
- Includes model weights, tokenizer, and config
- Automatic model loading with caching

### Updating Models
- Retrain periodically with new data
- Monitor performance degradation
- Version control for model artifacts