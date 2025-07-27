# LLM-Based Anomaly Detection

Language modeling approach for anomaly detection using fine-tuned transformer models to detect anomalous text patterns in data quality monitoring.

## Architecture Overview

The LLM-based anomaly detector uses **next-token probability prediction** to identify anomalies:

- **Normal text sequences** have high probability scores
- **Anomalous text sequences** have low probability scores

## System Design

### Training Phase
- Fine-tune a language model (DistilBERT) on clean text data from specific fields
- Use masked language modeling to learn probability distribution of normal patterns
- Save trained models for inference

### Detection Phase
- Calculate sequence probability for new text using trained models
- Compare against thresholds to determine anomalies
- Return anomaly details with probability scores and explanations

## Technical Architecture

### Model Components
- **Base Model**: DistilBERT (efficient transformer)
- **Training Method**: Masked Language Modeling (MLM)
- **Inference**: Next-token probability calculation
- **Thresholding**: Configurable probability thresholds

### Key Features
- **Field-specific models**: Each field gets its own trained language model
- **Probability-based detection**: Uses log probability scores
- **Context-aware**: Can incorporate temporal and categorical context
- **Interpretable**: Provides detailed explanations for anomalies

## Usage

### Training
```bash
python anomaly_detectors/llm_based/llm_model_training.py \
    data/training_data.csv \
    --field material \
    --epochs 3 \
    --threshold -2.0
```

### Detection
```python
from anomaly_detectors.llm_based.llm_anomaly_detector import LLMAnomalyDetector

# Create detector
detector = LLMAnomalyDetector(
    field_name="material",
    threshold=-2.0,
    use_gpu=True
)

# Load trained model
detector.learn_patterns(df, "material")

# Detect anomalies
anomalies = detector.bulk_detect(df, "material")
```

## Model Configuration

### Parameters
- `field_name`: Field to detect anomalies in
- `threshold`: Probability threshold for anomaly detection (default: -2.0)
- `use_gpu`: GPU acceleration (default: True)
- `epochs`: Training epochs (default: 2-3)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 2e-5)

## Suitable Fields

### Excellent Candidates
- **Text-rich fields**: Material compositions, care instructions, descriptions
- **Fields with rich vocabulary**: Multiple unique values and patterns

### Poor Candidates
- **Short fields**: Single words, codes
- **Structured data**: IDs, numbers
- **Limited vocabulary**: Fields with < 5 unique values

## Performance Characteristics

- **High recall**: Catches semantic anomalies effectively
- **GPU optimized**: Supports CUDA and Apple Silicon acceleration
- **Scalable**: Batch processing with configurable workers
- **Model caching**: Reuses loaded models for efficiency

## Integration

The LLM detector integrates with the unified detection system:
- Implements standard `bulk_detect()` interface
- Automatic model loading from saved checkpoints
- Graceful fallback when models unavailable
- Compatible with weighted combination system 