# LLM-Based Anomaly Detection

This module implements **language modeling for anomaly detection** using fine-tuned transformer models to detect anomalous text patterns in data quality monitoring.

## Overview

The LLM-based anomaly detector uses **next-token probability prediction** to identify anomalies in text fields. The approach is based on the principle that:

- **Normal text sequences** have high probability scores
- **Anomalous text sequences** have low probability scores

## How It Works

### 1. Training Phase
- **Fine-tune a language model** (e.g., DistilBERT) on clean text data from a specific field
- **Use masked language modeling** to learn the probability distribution of normal text patterns
- **Save the trained model** for inference

### 2. Detection Phase
- **Calculate sequence probability** for new text using the trained model
- **Compare against threshold** to determine if text is anomalous
- **Return anomaly details** with probability scores and explanations

## Key Features

### Language Modeling Approach
- **Field-specific models**: Each field gets its own trained language model
- **Probability-based detection**: Uses log probability scores for anomaly detection
- **Context-aware**: Can incorporate temporal and categorical context
- **Interpretable**: Provides detailed explanations for detected anomalies

### Model Architecture
- **Base Model**: DistilBERT (fast, efficient transformer)
- **Training**: Masked Language Modeling (MLM)
- **Inference**: Next-token probability calculation
- **Thresholding**: Configurable probability thresholds

### Optional Enhancements
- **Dynamic-aware encoding**: Incorporates temporal and contextual information
- **Prototype-based reprogramming**: Semantic alignment using learned prototypes
- **In-context learning**: Few-shot learning capabilities

## Usage

### Training a Model

```bash
# Train a language model for the material field
python anomaly_detectors/llm_based/llm_model_training.py \
    data/your_training_data.csv \
    --field material \
    --epochs 3 \
    --threshold -2.0
```

### Using in Demo

```bash
# Run demo with LLM detection enabled
python single_sample_multi_field_demo.py \
    --brand your_brand \
    --output-dir demo_llm_results \
    --enable-llm \
    --llm-threshold -2.0 \
    --core-fields-only
```

### Programmatic Usage

```python
from anomaly_detectors.llm_based.llm_anomaly_detector import LLMAnomalyDetector

# Create detector
detector = LLMAnomalyDetector(
    field_name="material",
    threshold=-2.0,
    use_gpu=True
)

# Learn patterns (loads trained model)
detector.learn_patterns(df, "material")

# Detect anomalies
anomalies = detector.bulk_detect(df, "material")
```

## Configuration

### Model Parameters
- `field_name`: Field to detect anomalies in (e.g., "material", "care_instructions")
- `threshold`: Probability threshold for anomaly detection (default: -2.0)
- `use_gpu`: Whether to use GPU acceleration (default: True)

### Training Parameters
- `epochs`: Number of training epochs (default: 2-3)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 2e-5)
- `mask_probability`: Probability of masking tokens during training (default: 0.15)

## Supported Fields

The LLM detector works best with **longer text fields** that have rich vocabulary:

### Excellent Candidates
- **Material**: `"95% Cotton - 3.5% Polyester - 1.5% Spandex"`
- **Care Instructions**: `"Machine wash cold, tumble dry low"`
- **Long Descriptions**: Product descriptions with natural language

### Poor Candidates
- **Short fields**: Color names, sizes, categories
- **Structured data**: EAN codes, article numbers
- **Limited vocabulary**: Fields with < 5 unique values

## Performance

### Training Results (Material Field)
- **Detection Rate**: 80% (4/5 obvious anomalies detected)
- **Probability Separation**: 2.018 difference between normal and anomalous
- **Training Time**: ~1.5 minutes on MPS (Apple Silicon GPU)

### Example Scores
- **Normal**: `"95% Cotton - 3.5% Polyester - 1.5% Spandex"` → -1.934
- **Anomalous**: `"INVALID MATERIAL TEXT"` → -7.344
- **Anomalous**: `"Random gibberish text"` → -4.737

## File Structure

```
anomaly_detectors/llm_based/
├── llm_anomaly_detector.py      # Main detector implementation
├── llm_model_training.py        # Training script
├── __init__.py                  # Module exports
└── README.md                    # This file

llm_results/                     # Trained models (generated)
├── material_model/              # Trained model for material field
├── care_instructions_model/     # Trained model for care instructions
└── *_training_results.json      # Training results and metrics
```

## Training Process

### 1. Data Analysis
- Analyzes unique values in the target field
- Checks field suitability for language modeling
- Reports vocabulary statistics

### 2. Model Training
- Loads pre-trained DistilBERT model
- Fine-tunes on field-specific text data
- Uses masked language modeling objective
- Saves fine-tuned model and tokenizer

### 3. Validation
- Tests model on obvious anomalies
- Calculates detection rates
- Reports probability separation metrics

## Integration

The LLM detector integrates seamlessly with the comprehensive detection system:

- **Compatible interface**: Implements `bulk_detect()` method
- **Automatic model loading**: Loads trained models on demand
- **Graceful fallback**: Skips detection for fields without trained models
- **Caching**: Reuses loaded models across multiple runs

## Advantages

### Over Traditional Methods
- **Semantic understanding**: Captures meaning, not just patterns
- **Context awareness**: Can use surrounding field information
- **Interpretable**: Provides probability scores and explanations
- **Scalable**: Can handle any text field with sufficient data

### Over Other LLM Approaches
- **Field-specific**: Tailored to each field's vocabulary and patterns
- **Efficient**: Uses smaller, faster models (DistilBERT)
- **Self-contained**: No external API dependencies
- **Offline**: Works without internet connection

## Limitations

- **Requires training data**: Needs clean data for each field
- **Field-specific models**: Each field needs its own trained model
- **Text fields only**: Works only with text-based data
- **Training time**: Requires time to fine-tune models

## Future Enhancements

- **Multi-field models**: Single model for multiple related fields
- **Active learning**: Continuous model improvement
- **Ensemble methods**: Combine multiple model predictions
- **Advanced context**: Better temporal and categorical integration 