# ML-Based Anomaly Detection System Documentation

## See Also

- [Main Project README](./README.md)
- [ML Anomaly Detection Overview](./ML_Anomaly_Detection_Overview.md)
- [Class Hierarchy Documentation](./CLASS_HIERARCHY_DOCUMENTATION.md)

## Overview

This documentation describes a comprehensive machine learning approach for anomaly detection in data quality monitoring, specifically designed for e-commerce product data. The system uses **SentenceTransformers** with **triplet loss** training to learn semantic representations that can distinguish between normal and anomalous data patterns.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Training Approach](#training-approach)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Anomaly Detection Methods](#anomaly-detection-methods)
5. [Implementation Details](#implementation-details)
6. [Configuration and Setup](#configuration-and-setup)
7. [Usage Examples](#usage-examples)
8. [Performance Analysis](#performance-analysis)
9. [Future Improvements](#future-improvements)

## System Architecture

### Core Components

The system consists of several interconnected modules:

```
anomaly_detectors/ml/
├── index.py                     # Main CLI interface
├── model_training.py            # Training logic and evaluation
├── hyperparameter_search.py     # HP optimization
├── check_anomalies.py          # Inference and detection
├── rule_column_map.py          # Column mapping configuration
├── optimal_params.json         # Best parameters per column
├── hyperparameter_search_space.json  # Search space definition
└── requirements.txt            # Dependencies
```

### Design Philosophy

The system is built around a **"recall-focused"** approach, prioritizing the detection of actual anomalies over minimizing false positives. This is particularly important for data quality monitoring where missing real issues is more costly than flagging some normal data for review.

**Key Design Principle**: The system operates under the realistic constraint that **clean reference data is never available during production anomaly detection**. All detection methods must work solely based on the patterns learned during training, without requiring external clean examples for comparison.

## Training Approach

### 1. Triplet Loss Architecture

The system uses **triplet loss** to train SentenceTransformer models:

- **Anchor**: Clean, valid data example
- **Positive**: Another clean, valid data example (similar to anchor)
- **Negative**: Error-injected or corrupted version of the data (anomaly)

```python
# Example triplet structure for material data
Anchor:   "95% Cotton - 5% Spandex"
Positive: "100% Cotton"  
Negative: "95% Cottn - 5% Spandex"  # Typo injection
```

### 2. Error Injection Strategy

The training process uses sophisticated error injection rules to create realistic anomalies:

#### Material Column Error Types:
- **Unicode errors**: `é` → `Ãª`
- **Composition errors**: `100%` → `99%` (sum doesn't equal 100%)
- **Missing symbols**: Remove percentage signs
- **Invalid characters**: Add trademark symbols `Cotton™`
- **Delimiter errors**: Replace commas with semicolons
- **Format corruption**: `"100% Cotton"` → `"cotton 100"`
- **Spacing issues**: Multiple spaces, line breaks
- **Random noise**: Character duplication, random insertions

### 3. Text Preprocessing

Robust preprocessing ensures consistent model training:

```python
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text
```

### 4. Triplet Dataset Creation

The system generates balanced triplet datasets:

```python
def create_improved_triplet_dataset(data_series, rules, column_name):
    """
    Creates triplets where:
    - All clean values are considered similar to each other
    - Anomalies (error-injected) are distant from clean values
    """
    for anchor_text in clean_texts:
        for positive_text in clean_texts:
            if positive_text != anchor_text:
                # Apply random error rule to create negative example
                negative_text = apply_error_rule(anchor_text, random.choice(rules))
                
                if negative_text != anchor_text:  # Ensure corruption occurred
                    triplet = InputExample(
                        texts=[anchor_text, positive_text, negative_text],
                        label=0  # Not used in triplet loss
                    )
                    triplets.append(triplet)
```

## Hyperparameter Optimization

### 1. Search Strategy

The system implements **recall-focused hyperparameter search** with precision constraints:

- **Primary objective**: Maximize recall (detect as many anomalies as possible)
- **Constraint**: Maintain minimum 30% precision to avoid excessive false positives
- **Search method**: Random search across predefined parameter space

### 2. Search Space Configuration

Defined in `hyperparameter_search_space.json`:

```json
{
  "model_name": [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
  ],
  "triplet_margin": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
  "distance_metric": ["COSINE", "EUCLIDEAN", "MANHATTAN"],
  "batch_size": [8, 16, 24, 32, 48, 64],
  "epochs": [2, 3, 4, 5, 6, 8],
  "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
}
```

### 3. Performance Evaluation

Each hyperparameter combination is evaluated using:

```python
def evaluate_recall_and_precision_performance(model, clean_texts, rules, column_name):
    # Test recall: Create anomalies and check detection rate
    for clean_text in test_samples:
        corrupted_text = apply_error_rule(clean_text, random.choice(rules))
        similarity = cosine_similarity(
            model.encode([clean_text]), 
            model.encode([corrupted_text])
        )[0][0]
        
        if similarity < threshold:  # Anomaly detected
            detected_anomalies += 1
    
    recall = detected_anomalies / total_anomalies
    
    # Test precision: Check clean-to-clean similarities for false positives
    # ... similar logic for precision calculation
```

### 4. Optimal Parameters

Best parameters are stored per column in `optimal_params.json`:

```json
{
  "material": {
    "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "triplet_margin": 1.0,
    "distance_metric": "COSINE", 
    "batch_size": 48,
    "epochs": 3,
    "learning_rate": 5e-6
  }
}
```

## Anomaly Detection Methods

### Production Detection Algorithm

The anomaly detection uses the model's learned representation **without requiring any clean reference data**. This is the only method used in production since clean reference data is never available during real-world inference:

```python
def check_anomalies(model, values, threshold=0.6):
    """
    Detects anomalies using statistical analysis of embeddings:
    1. Compute embeddings for all values in the dataset
    2. Calculate centroid as "normal" baseline from the data itself
    3. Identify outliers using both global and local similarity measures
    
    Key insight: The model learned during training with triplet loss can 
    distinguish normal from anomalous patterns without external references.
    """
    
    # Get embeddings for all values
    embeddings = model.encode(processed_values)
    
    # Compute centroid of all embeddings as baseline "normal" representation
    centroid = np.mean(embeddings, axis=0)
    
    # Calculate similarities to centroid
    centroid_sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # Calculate pairwise similarities
    pairwise_sims = cosine_similarity(embeddings)
    
    for i, value in enumerate(values):
        # Similarity to global centroid
        centroid_sim = float(centroid_sims[i])
        
        # Average similarity to other values (local neighborhood)
        other_sims = np.concatenate([pairwise_sims[i][:i], pairwise_sims[i][i+1:]])
        avg_sim_to_others = float(np.mean(other_sims))
        
        # Use minimum of both measures (catches both global and local outliers)
        final_similarity = min(centroid_sim, avg_sim_to_others)
        
        is_anomaly = final_similarity < threshold
```

### Detection Logic Explained

The algorithm works in two phases:

1. **Global Outlier Detection**: Compares each value to the centroid of all embeddings
   - Values far from the centroid are globally unusual
   - Catches obvious anomalies like completely different data types

2. **Local Outlier Detection**: Compares each value to its neighbors in the dataset
   - Values dissimilar to nearby data points are locally unusual  
   - Catches subtle anomalies that might be close to the global centroid

3. **Conservative Decision**: Takes the minimum similarity score
   - If a value is unusual either globally OR locally, flag it as anomaly
   - Prioritizes recall (catching anomalies) over precision (avoiding false positives)

### Why This Works Without Clean References

The triplet loss training teaches the model that:
- **Normal values** have similar embeddings to each other
- **Anomalous values** have dissimilar embeddings from normal values

During inference, truly anomalous values will:
- Be distant from the centroid of mostly-normal data
- Be dissimilar to their neighboring values in the dataset
- Stand out statistically from the pattern of normal embeddings

### Threshold Selection

The system uses different thresholds based on the use case:
- **Default threshold**: 0.6 (balanced detection)
- **Strict threshold**: 0.95+ (catch subtle anomalies)
- **Lenient threshold**: 0.3-0.5 (only obvious anomalies)

**Critical Design Constraint**: All threshold tuning and detection methods are designed to work **without any clean reference data** during production use. The model must rely entirely on patterns learned during training to distinguish normal from anomalous data.

## Implementation Details

### 1. Column-Specific Configuration

Each data column has optimized configurations:

```python
def get_column_configs():
    return {
        'material': {
            'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 
            'epochs': 3
        },
        'colour_name': {
            'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 
            'epochs': 4
        },
        'article_number': {
            'model': 'sentence-transformers/all-mpnet-base-v2', 
            'epochs': 5
        }
    }
```

### 2. Rule-to-Column Mapping

The system maps validation rule names to actual CSV column names:

```python
def get_rule_to_column_map():
    return {
        "material": "material",
        "color_name": "colour_name", 
        "size": "size_name",
        "article_number": "article_number"
    }
```

### 3. Model Architecture

Trained models use standardized SentenceTransformer architecture:

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_mean_tokens': True})
  (2): Normalize()
)
```

### 4. Training Pipeline

```python
def train_and_evaluate_similarity_model(df, column, rules, device, best_params):
    # 1. Create triplet dataset
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    
    # 2. Initialize model with optimal parameters
    model = SentenceTransformer(best_params['model_name'])
    
    # 3. Configure triplet loss
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=best_params['distance_metric'],
        triplet_margin=best_params['triplet_margin']
    )
    
    # 4. Train model
    model.fit(
        train_objectives=[(DataLoader(triplets, batch_size=best_params['batch_size']), train_loss)],
        epochs=best_params['epochs'],
        warmup_steps=50,
        output_path=model_results_dir,
        save_best_model=True
    )
```

## Configuration and Setup

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies:
# sentence-transformers>=2.2.0
# torch>=1.12.0
# pandas>=1.5.0
# scikit-learn>=1.1.0
```

### 2. Directory Structure

```
anomaly_detectors/
├── ml/                          # Main ML implementation
├── results/                     # Trained models
│   ├── results_material/
│   ├── results_colour_name/
│   └── ...
└── cache/                       # Cached data samples
```

### 3. GPU Support

The system automatically detects and uses available hardware:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple M1/M2
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU fallback
```

## Usage Examples

### 1. Training Models

Train models for all columns:
```bash
python index.py ../../data/esqualo_2022_fall_original.csv
```

Train with hyperparameter search:
```bash
python index.py ../../data/esqualo_2022_fall_original.csv --use-hp-search --hp-trials 20
```

Train specific columns only:
```bash
python index.py ../../data/esqualo_2022_fall_original.csv --rules material color_name
```

### 2. Running Anomaly Detection

Basic anomaly detection:
```bash
python index.py ../../data/esqualo_2022_fall.csv --check-anomalies material --output results.csv
```

With custom threshold:
```bash
python index.py ../../data/esqualo_2022_fall.csv --check-anomalies material --threshold 0.95 --output results.csv
```

### 3. Programmatic Usage

```python
from check_anomalies import load_model_for_rule, check_anomalies
import pandas as pd

# Load trained model
model, column_name = load_model_for_rule("material", results_dir="../results")

# Load test data  
df = pd.read_csv("test_data.csv")
values = df[column_name].tolist()

# Run detection
results = check_anomalies(model, values, threshold=0.6)

# Process results
anomalies = [r for r in results if r['is_anomaly']]
print(f"Found {len(anomalies)} anomalies")
```

## Performance Analysis

### 1. Model Performance by Column

Based on hyperparameter search results:

| Column | Model | Recall | Precision | F1-Score |
|--------|-------|--------|-----------|----------|
| material | multi-qa-MiniLM-L6-cos-v1 | 0.882 | 0.65 | 0.748 |
| colour_name | multi-qa-MiniLM-L6-cos-v1 | 1.0 | 0.85 | 0.919 |
| colour_code | all-mpnet-base-v2 | 1.0 | 0.90 | 0.947 |
| article_structure_name_2 | multi-qa-MiniLM-L6-cos-v1 | 1.0 | 0.92 | 0.958 |

### 2. Threshold Impact Analysis

Results for material column with different thresholds:

| Threshold | Anomalies Detected | Types Caught |
|-----------|-------------------|--------------|
| 0.1 | 0 | None (too strict) |
| 0.5 | 4 | Obvious anomalies ("amin", empty values) |
| 0.97 | 18 | Includes subtle pattern deviations |

### 3. Detection Capabilities

**Successfully Detected:**
- Completely invalid values ("amin" in material field)
- Empty/null values
- Unusual composition patterns ("99% Cotton - 1% Viscose")
- Severe formatting errors

**Limitations:**
- Subtle typos that maintain overall pattern ("95% Cottn...")
- Business rule violations (percentages > 100%) that follow format
- Context-dependent errors requiring domain knowledge

## Future Improvements

### 1. Enhanced Training Data

- **Synthetic data generation**: Create more diverse error patterns
- **Real-world anomaly collection**: Gather actual problematic data
- **Multi-language support**: Train on international product data
- **Temporal adaptation**: Update models with new data patterns

### 2. Advanced Detection Methods

- **Ensemble approaches**: Combine multiple models and thresholds
- **Context-aware detection**: Use surrounding column values as context
- **Business rule integration**: Combine ML with deterministic validation
- **Confidence calibration**: Better uncertainty estimation

### 3. Operational Enhancements

- **Real-time inference**: Optimize for streaming data processing
- **Model versioning**: Track and manage model updates
- **A/B testing framework**: Compare detection approaches
- **Feedback loop**: Learn from user corrections

### 4. Explainability Features

- **Attention visualization**: Show which parts of text drive decisions
- **Nearest neighbor explanations**: Compare to similar normal examples
- **Pattern-based explanations**: Map ML decisions to human-readable rules
- **Confidence intervals**: Provide uncertainty estimates

### 5. Scalability Improvements

- **Distributed training**: Scale to larger datasets
- **Model compression**: Reduce inference time and memory usage
- **Batch processing**: Optimize for large-scale data processing
- **Edge deployment**: Run models on resource-constrained devices

## Conclusion

This ML-based anomaly detection system provides a robust, scalable approach to data quality monitoring. By leveraging modern NLP techniques and focusing on recall optimization, it successfully identifies a wide range of data quality issues while maintaining practical precision levels.

The system's modular design allows for easy extension to new data types and domains, while the comprehensive evaluation framework ensures reliable performance across different scenarios. The combination of automated hyperparameter optimization and manual configuration provides both convenience and control for different use cases.

Key strengths include:
- **High recall**: Catches most anomalies through semantic understanding
- **No reference data required**: Works without clean examples during production inference
- **Column-specific optimization**: Tailored approaches for different data types
- **Comprehensive evaluation**: Rigorous testing and validation methodology
- **Real-world viability**: Designed for scenarios where clean reference data is unavailable

The system represents a significant advancement in automated data quality monitoring, providing organizations with powerful tools to maintain data integrity at scale.
