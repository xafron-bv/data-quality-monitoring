# ML-Based Anomaly Detection

Semantic similarity-based anomaly detection using sentence transformers.

## Overview

This module uses machine learning to detect anomalies by learning semantic patterns from clean data. It identifies values that are semantically different from the expected patterns.

## How It Works

1. **Training Phase**:
   - Fine-tunes sentence transformer models on field-specific data
   - Creates triplet datasets (anchor, positive, negative examples)
   - Generates reference centroids from clean data

2. **Detection Phase**:
   - Encodes input values into embeddings
   - Calculates cosine similarity to reference centroid
   - Flags values below similarity threshold as anomalies

## Key Components

### Core Files
- `ml_anomaly_detector.py`: Main detector implementation with caching
- `model_training.py`: Training script for creating field-specific models
- `check_anomalies.py`: Anomaly detection logic and centroid comparison
- `ml_anomaly_reporter.py`: Formats ML detection results

### Utilities
- `gpu_utils.py`: GPU detection and optimization
- `index.py`: Model indexing utilities
- `generate_centroids_for_existing_models.py`: Centroid generation tool

### Configuration
- `optimal_params.json`: Best hyperparameters per field
- `hyperparameter_search_space.json`: Search space definition
- `ml_explanation_templates.json`: Human-readable explanations

## Training a New Model

### 1. Prepare Data
Ensure you have clean data samples for the field you want to train.

### 2. Configure Field
Add field configuration in `model_training.py`:
```python
'new_field': {
    'model': 'sentence-transformers/all-MiniLM-L6-v2',
    'epochs': 3
}
```

### 3. Run Training
```bash
python ../../main.py ml-train your_data.csv \
    --fields new_field
```

### 4. Model Output
Models are saved to `models/trained/{field_name}/`:
- `config.json`: Model configuration
- `model.safetensors`: Model weights
- `reference_centroid.npy`: Reference embedding

## Performance by Field

Based on hyperparameter search results:

### Excellent (1.0 recall)
- `category`: multi-qa-MiniLM-L6-cos-v1, 4 epochs
- `colour_code`: all-mpnet-base-v2, 2 epochs
- `color_name`: multi-qa-MiniLM-L6-cos-v1, 4 epochs

### Good (0.8+ recall)
- `material`: multi-qa-MiniLM-L6-cos-v1, 3 epochs

### Moderate (0.4-0.5 recall)
- `ean`: multi-qa-MiniLM-L6-cos-v1, 3 epochs
- `size`: paraphrase-MiniLM-L6-v2, 8 epochs

## Hyperparameter Tuning

Run hyperparameter search:
```bash
python ../../main.py ml-train your_data.csv \
    --fields material \
    --use-hp-search \
    --hp-trials 50
```

## GPU Acceleration

The module automatically detects and uses GPU if available:
- CUDA devices preferred
- Falls back to CPU if no GPU
- Configurable via `use_gpu` parameter

## Memory Optimization

- Models are cached at class level
- Reference centroids loaded once
- Batch processing for large datasets
- Automatic garbage collection

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use CPU instead of GPU
- Process fields sequentially

### Poor Detection Performance
- Check training data quality
- Adjust similarity threshold
- Try different sentence transformer model
- Increase training epochs