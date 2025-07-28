# ML-Based Anomaly Detection

## Overview

The ML-based detection layer uses deep learning embeddings to identify semantic anomalies in data. It employs Sentence Transformers to convert text values into high-dimensional vectors and uses centroid-based anomaly detection to identify outliers.

## Architecture

### Core Components

1. **MLAnomalyDetector** (`ml_anomaly_detector.py`)
   - Implements `AnomalyDetectorInterface`
   - Manages model loading and caching
   - Performs embedding generation and similarity computation

2. **Model Training** (`model_training.py`)
   - Trains field-specific Sentence Transformer models
   - Generates reference centroids from clean data
   - Optimizes for domain-specific anomaly detection

3. **GPU Utilities** (`gpu_utils.py`)
   - Automatic GPU detection and allocation
   - Fallback to CPU when GPU unavailable
   - Memory management for large batches

## Detection Process

### 1. Embedding Generation
```python
# Text preprocessing
preprocessed_text = preprocess_text(value)

# Generate embedding
embedding = model.encode(preprocessed_text)
```

### 2. Centroid Comparison
```python
# Compute similarity to reference centroid
similarity = cosine_similarity(embedding, centroid)

# Anomaly score = 1 - similarity
anomaly_score = 1.0 - similarity
```

### 3. Threshold-Based Detection
- Values with similarity below threshold are flagged as anomalies
- Threshold is field-specific and tunable
- Default threshold: 0.7 (30% deviation from centroid)

## Model Architecture

### Sentence Transformers
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Max Sequence Length**: 256 tokens
- **Pooling**: Mean pooling over token embeddings

### Fine-Tuning Strategy
1. **Contrastive Learning**: Similar values pushed together, dissimilar apart
2. **Domain Adaptation**: Fine-tuned on field-specific data
3. **Centroid Learning**: Reference centroids computed from clean samples

## Field-Specific Models

Each field has its own trained model stored in `results/`:

```
results/
├── material/
│   ├── model/           # Fine-tuned Sentence Transformer
│   ├── centroid.npy     # Reference centroid
│   └── config.json      # Model configuration
├── category/
├── color_name/
└── ...
```

## Training Process

### 1. Data Preparation
```bash
python model_training.py \
  --data-file data/training_data.csv \
  --field material \
  --validation-split 0.2
```

### 2. Model Fine-Tuning
- **Epochs**: 10-20 (early stopping based on validation loss)
- **Batch Size**: 32-64 (depending on GPU memory)
- **Learning Rate**: 2e-5 with warmup
- **Loss Function**: Contrastive loss or triplet loss

### 3. Centroid Generation
```python
# Generate embeddings for clean samples
embeddings = model.encode(clean_samples)

# Compute centroid
centroid = np.mean(embeddings, axis=0)
```

## Anomaly Scoring

### Similarity Metrics
- **Cosine Similarity**: Default metric, scale-invariant
- **Euclidean Distance**: Alternative for absolute differences
- **Mahalanobis Distance**: Considers covariance (experimental)

### Score Interpretation
- **0.0-0.3**: Normal values (high similarity to centroid)
- **0.3-0.5**: Borderline anomalies
- **0.5-0.7**: Likely anomalies
- **0.7-1.0**: Strong anomalies

## Performance Optimization

### GPU Acceleration
```python
# Automatic GPU selection
device = get_optimal_device(use_gpu=True)
model.to(device)
```

### Batch Processing
```python
# Process in batches for efficiency
for batch in batch_iterator(values, batch_size=1000):
    embeddings = model.encode(batch, batch_size=32)
```

### Model Caching
- Models cached in memory after first load
- Shared across detector instances
- Automatic cache management based on memory

## Hyperparameter Tuning

### Automated Search
```bash
python hyperparameter_search.py \
  --field material \
  --trials 50 \
  --metric f1_score
```

### Key Hyperparameters
- **similarity_threshold**: Anomaly detection threshold
- **embedding_batch_size**: Batch size for encoding
- **preprocessing_level**: Text normalization intensity
- **centroid_samples**: Number of samples for centroid

## Integration Features

### Explanation Generation
ML detector provides human-readable explanations:
```python
"Material 'Polyester 50% Coton 50%' deviates significantly from typical patterns (confidence: 0.85)"
```

### Confidence Calibration
- Scores calibrated using validation data
- Field-specific calibration curves
- Ensures consistent confidence interpretation

## Advantages

1. **Semantic Understanding**: Captures meaning beyond surface patterns
2. **Adaptability**: Learns from data without explicit rules
3. **Language Agnostic**: Works across different languages
4. **Continuous Learning**: Can be retrained with new data

## Limitations

1. **Black Box**: Less interpretable than rule-based methods
2. **Data Dependency**: Requires quality training data
3. **Computational Cost**: Higher than pattern-based detection
4. **Cold Start**: Needs initial training before use

## Best Practices

1. **Regular Retraining**: Update models with new clean data
2. **Threshold Tuning**: Adjust thresholds based on use case
3. **Ensemble Approach**: Combine with other detection methods
4. **Monitoring**: Track model performance over time
5. **Explainability**: Always provide confidence scores and explanations