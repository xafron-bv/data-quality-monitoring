# LLM-Based Anomaly Detection

Advanced anomaly detection using language models with contextual understanding.

## Overview

This module leverages large language models (LLMs) for sophisticated anomaly detection with:
- Dynamic context encoding
- Few-shot learning capabilities
- Temporal awareness
- Prototype-based reprogramming

## Key Features

### 1. Dynamic Context Encoding
Incorporates contextual information beyond the field value:
- Temporal data (dates, seasons)
- Related fields (category, brand)
- Historical patterns

### 2. Few-Shot Learning
Can adapt to new patterns with minimal examples:
```python
detector = create_llm_detector_for_field(
    field_name="material",
    few_shot_examples=["100% Cotton", "Polyester Blend"]
)
```

### 3. Prototype-Based Reprogramming
Learns semantic prototypes from data clusters to improve detection accuracy.

## Architecture

### Core Components
- `llm_anomaly_detector.py`: Main detector with advanced features
- `llm_model_training.py`: Fine-tuning script for LLMs
- `DynamicAwareEncoder`: Handles temporal and categorical context
- `PrototypeBasedReprogramming`: Semantic alignment using prototypes
- `InContextLearningDetector`: Few-shot learning implementation

## Training Process

### 1. Prepare Training Data
```python
# Format: DataFrame with labeled anomalies
train_df = pd.DataFrame({
    'material': ['Cotton', 'Poly3ster', 'Wool'],
    'is_anomaly': [0, 1, 0]
})
```

### 2. Configure Context (Optional)
```bash
python llm_model_training.py \
    --field material \
    --temporal-column date \
    --context-columns category,brand
```

### 3. Run Training
```bash
python llm_model_training.py \
    --field material \
    --train-file train_data.csv \
    --val-file val_data.csv \
    --epochs 10
```

## Usage Examples

### Basic Detection
```python
detector = create_llm_detector_for_field("material")
detector.learn_patterns(df, "Material_Column")
anomalies = detector.bulk_detect(df, "Material_Column")
```

### With Context
```python
detector = create_llm_detector_for_field(
    field_name="material",
    temporal_column="season",
    context_columns=["category", "brand"]
)
```

### Few-Shot Learning
```python
detector = create_llm_detector_for_field(
    field_name="color_name",
    few_shot_examples=[
        "Royal Blue",
        "Forest Green",
        "Crimson Red"
    ]
)
```

## Configuration Options

### Model Selection
- Default: `distilbert-base-uncased`
- Options: Any HuggingFace transformer model

### Thresholds
- `threshold`: Anomaly detection sensitivity (default: 0.6)
- `dynamic_threshold`: Enable adaptive thresholding

### Context Settings
- `temporal_column`: Column with time-based data
- `context_columns`: List of related columns
- `enable_prototypes`: Use prototype learning

## Performance Optimization

### Memory Management
- Model quantization available
- Batch processing with configurable size
- Automatic cleanup after detection

### Speed Optimization
- GPU acceleration when available
- Cached tokenization
- Parallel processing support

## Advanced Features

### Temporal Awareness
```python
# Detects seasonal anomalies
detector = create_llm_detector_for_field(
    "material",
    temporal_column="season",
    enable_temporal_encoding=True
)
```

### Multi-Field Context
```python
# Uses category context for material validation
detector = create_llm_detector_for_field(
    "material",
    context_columns=["category", "sub_category"],
    cross_field_attention=True
)
```

## Best Practices

1. **Training Data Quality**: Ensure balanced, representative training sets
2. **Context Selection**: Choose relevant context columns
3. **Threshold Tuning**: Adjust based on precision/recall requirements
4. **Regular Retraining**: Update models with new patterns

## Troubleshooting

### High False Positive Rate
- Lower detection threshold
- Add more training examples
- Enable few-shot learning

### Slow Performance
- Reduce batch size
- Use smaller model (e.g., DistilBERT)
- Enable GPU acceleration

### Context Not Working
- Verify column names
- Check data types
- Ensure context columns have values