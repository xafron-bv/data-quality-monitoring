# LLM-Based Anomaly Detection

## Overview

The LLM-based detection layer leverages fine-tuned language models for context-aware anomaly detection. It combines the semantic understanding of transformer models with dynamic encoding techniques to identify subtle anomalies that rule-based and embedding methods might miss.

## Architecture

### Core Components

1. **LLMAnomalyDetector** (`llm_anomaly_detector.py`)
   - Fine-tuned transformer models for anomaly classification
   - Dynamic-aware encoding with temporal and contextual features
   - Few-shot learning capabilities

2. **Model Training** (`llm_model_training.py`)
   - Fine-tunes pre-trained language models on field-specific data
   - Implements custom loss functions for anomaly detection
   - Supports both classification and similarity-based approaches

3. **Dynamic Context Encoding**
   - Incorporates temporal information (seasons, dates)
   - Considers categorical context (product type, brand)
   - Adapts to domain-specific patterns

## Detection Approaches

### 1. Classification-Based Detection
```python
# Fine-tuned classifier approach
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # Normal vs Anomaly
)

# Predict anomaly probability
outputs = model(encoded_text)
anomaly_prob = torch.softmax(outputs.logits, dim=1)[:, 1]
```

### 2. Similarity-Based Detection
```python
# Masked language model approach
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Compute perplexity as anomaly score
perplexity = compute_perplexity(model, text)
anomaly_score = normalize_perplexity(perplexity)
```

### 3. Few-Shot Learning
```python
# Dynamic few-shot examples
examples = get_similar_examples(value, k=5)
prompt = create_few_shot_prompt(examples, value)
anomaly_score = model.predict(prompt)
```

## Dynamic Context Integration

### Temporal Encoding
```python
@dataclass
class DynamicContext:
    temporal_info: float  # Normalized time feature
    categorical_info: str  # Product category
    contextual_info: Dict[str, Any]  # Additional context
```

### Context-Aware Features
1. **Seasonal Patterns**: Fashion trends by season
2. **Category Relationships**: Material expectations by product type
3. **Brand Specifics**: Brand-specific naming conventions
4. **Historical Trends**: Evolution of product attributes

## Model Architecture

### Base Models
- **BERT**: For general text understanding
- **RoBERTa**: For robust performance
- **DistilBERT**: For faster inference
- **Custom Models**: Domain-specific pre-training

### Fine-Tuning Strategy

1. **Data Augmentation**
   ```python
   # Generate synthetic anomalies
   augmented_data = create_synthetic_anomalies(
       clean_data,
       error_types=['typos', 'format_errors', 'semantic_shifts']
   )
   ```

2. **Contrastive Learning**
   ```python
   # Learn to distinguish normal from anomalous
   loss = contrastive_loss(
       anchor=clean_sample,
       positive=similar_clean,
       negative=anomaly_sample
   )
   ```

3. **Multi-Task Learning**
   - Primary task: Anomaly detection
   - Auxiliary tasks: Field type classification, language detection

## Training Process

### 1. Data Preparation
```bash
python llm_model_training.py \
  --field material \
  --model-name bert-base-uncased \
  --train-data data/train.csv \
  --epochs 5
```

### 2. Training Configuration
```python
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="steps"
)
```

### 3. Model Evaluation
- **Metrics**: F1, Precision, Recall, AUC-ROC
- **Validation**: K-fold cross-validation
- **Calibration**: Probability calibration for confidence scores

## Advanced Features

### 1. Explainable Predictions
```python
# Attention-based explanations
attention_weights = model.get_attention_weights(text)
important_tokens = extract_important_tokens(attention_weights)

explanation = f"Anomaly detected due to unusual pattern in: {important_tokens}"
```

### 2. Adaptive Thresholds
```python
# Dynamic threshold based on context
threshold = base_threshold * context_modifier * confidence_factor
```

### 3. Ensemble Methods
```python
# Combine multiple LLM predictions
predictions = [
    bert_model.predict(text),
    roberta_model.predict(text),
    custom_model.predict(text)
]
final_score = weighted_average(predictions, weights)
```

## Performance Optimization

### 1. Model Quantization
```python
# 8-bit quantization for faster inference
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 2. Caching Strategies
- **Embedding Cache**: Store computed embeddings
- **Prediction Cache**: Cache common predictions
- **Context Cache**: Reuse computed contexts

### 3. Batch Processing
```python
# Efficient batch inference
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=32):
        predictions = model(batch)
```

## Integration with Other Methods

### Weighted Ensemble
```python
final_score = (
    llm_score * 0.4 +
    ml_score * 0.3 +
    pattern_score * 0.2 +
    validation_score * 0.1
)
```

### Hierarchical Detection
1. Fast pattern-based filtering
2. ML detection for borderline cases
3. LLM for complex anomalies

## Use Cases

### 1. Semantic Anomalies
- "Cotton 100%" vs "100% Cotton" (format variation)
- "Polyester" vs "Plyester" (typo detection)
- "Silk blend" vs "Silk mixture" (synonym handling)

### 2. Context-Dependent Anomalies
- "Wool" in summer collection (seasonal mismatch)
- "Leather" in vegan product line (category conflict)
- "Waterproof cotton" (material property conflict)

### 3. Complex Patterns
- Mixed language descriptions
- Inconsistent formatting within same field
- Subtle semantic shifts over time

## Best Practices

1. **Model Selection**: Choose model size based on latency requirements
2. **Fine-Tuning Data**: Ensure balanced representation of anomaly types
3. **Threshold Tuning**: Validate thresholds on holdout data
4. **Regular Updates**: Retrain models with new patterns
5. **Explainability**: Always provide reasoning for detections
6. **Resource Management**: Monitor GPU/CPU usage
7. **Fallback Strategy**: Have backup detection when LLM fails

## Limitations

1. **Computational Cost**: Higher than other methods
2. **Latency**: Slower inference than rule-based
3. **Interpretability**: Complex to explain all decisions
4. **Data Requirements**: Needs substantial training data
5. **Drift**: Model performance may degrade over time