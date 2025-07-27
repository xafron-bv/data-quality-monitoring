# ML Anomaly Detection System - Technical Architecture

## System Overview

A production-grade anomaly detection pipeline using **SentenceTransformers** with **triplet loss** training for semantic data quality monitoring. Designed for zero-dependency inference where clean reference data is unavailable.

## Core Architecture

### Training Pipeline
```
Clean Data → Error Injection → Triplet Generation → Model Training → Hyperparameter Optimization
```

**Key Design Decisions:**
- **Triplet Loss**: Anchor (clean) + Positive (clean) + Negative (synthetic error)
- **Semantic Embeddings**: 384-dim dense vectors from pre-trained transformers
- **Error Synthesis**: Rule-based corruption engine with 14+ error types per domain
- **Recall Optimization**: Prioritize anomaly detection over false positive reduction

### Inference Architecture
```
Raw Data → Text Preprocessing → Model Encoding → Statistical Outlier Detection → Anomaly Scoring
```

**Critical Constraint**: No clean reference data available during production inference.

## Technical Implementation

### Model Architecture
```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'architecture': 'BertModel'})
  (1): Pooling({'pooling_mode_mean_tokens': True, 'dimension': 384})
  (2): Normalize()
)
```

### Anomaly Detection Algorithm
```python
def detect_anomalies(embeddings, threshold=0.6):
    # Global outlier detection via centroid analysis
    centroid = np.mean(embeddings, axis=0)
    global_sims = cosine_similarity(embeddings, centroid.reshape(1, -1))
    
    # Local outlier detection via neighborhood analysis  
    pairwise_sims = cosine_similarity(embeddings)
    local_sims = np.mean(pairwise_sims, axis=1)
    
    # Conservative decision: flag if unusual globally OR locally
    final_scores = np.minimum(global_sims.flatten(), local_sims)
    return final_scores < threshold
```

## Technical Features

### Self-Supervised Learning
- **Contrastive Training**: Learn semantic boundaries between normal/anomalous patterns
- **Domain Adaptation**: Column-specific model fine-tuning with optimized hyperparameters
- **Transfer Learning**: Leverage pre-trained language model representations

### Synthetic Data Generation
- **Error Injection Engine**: Configurable rule-based corruption with realistic error patterns
- **Balanced Sampling**: Maintain positive/negative ratios during triplet generation
- **Pattern Preservation**: Ensure synthetic errors maintain domain-specific characteristics

### Reference-Free Inference
- **Statistical Outlier Detection**: Combine global (centroid-based) and local (neighborhood-based) analysis
- **Embedding Space Analysis**: Leverage learned semantic representations without external baselines
- **Threshold Calibration**: Configurable sensitivity based on operational requirements

## Performance Characteristics

### Model Performance by Domain
| Domain | Architecture | Embedding Dim | Recall | Precision | F1 |
|--------|-------------|---------------|--------|-----------|-----|
| Material Composition | multi-qa-MiniLM-L6-cos-v1 | 384 | 0.882 | 0.65 | 0.748 |
| Color Names | multi-qa-MiniLM-L6-cos-v1 | 384 | 1.0 | 0.85 | 0.919 |
| Product Categories | multi-qa-MiniLM-L6-cos-v1 | 384 | 1.0 | 0.92 | 0.958 |
| Color Codes | all-mpnet-base-v2 | 768 | 1.0 | 0.90 | 0.947 |

### Hyperparameter Optimization
- **Search Strategy**: Random search with recall-focused objective + precision constraint (≥30%)
- **Search Space**: 6D parameter space (model, margin, distance metric, batch size, epochs, LR)
- **Evaluation**: Cross-validation with synthetic anomaly injection
- **Convergence**: 15-20 trials typically sufficient for optimal configuration

## System Architecture

### Directory Structure
```
anomaly_detectors/ml_based/
├── ml_anomaly_detector.py      # Main detector implementation
├── model_training.py           # Triplet loss training pipeline
├── hyperparameter_search.py    # Bayesian/random HP optimization
├── check_anomalies.py         # Reference-free inference engine
├── rule_column_map.py         # Domain-specific configuration
├── optimal_params.json        # Pre-computed optimal hyperparameters
└── requirements.txt           # Dependencies (torch, transformers, sklearn)
```

### Configuration Management
```python
# Column-specific model configurations
COLUMN_CONFIGS = {
    'material': {
        'model_name': 'multi-qa-MiniLM-L6-cos-v1',
        'triplet_margin': 1.0,
        'distance_metric': 'COSINE',
        'batch_size': 48,
        'epochs': 3,
        'learning_rate': 5e-6
    }
}
```

## Deployment & Operations

### Training Workflow
```bash
# Full pipeline with hyperparameter optimization
python anomaly_detectors/ml_based/model_training.py material data.csv --use-hp-search --hp-trials 20

# Production training with pre-optimized parameters  
python anomaly_detectors/ml_based/model_training.py material data.csv
```

### Inference Workflow
```bash
# Standard anomaly detection
python anomaly_detectors/ml_based/check_anomalies.py data.csv --field material --threshold 0.6
```

### Hardware Optimization
- **GPU Acceleration**: Auto-detection (CUDA/MPS/CPU fallback)
- **Batch Processing**: Configurable batch sizes for memory optimization
- **Model Caching**: Persistent model storage for rapid inference startup

## Technical Innovations

### Triplet Loss Design
- **Semantic Similarity Learning**: Train models to cluster similar concepts while separating anomalies
- **Error-Aware Training**: Use domain-specific error injection for realistic negative examples
- **Multi-Scale Detection**: Combine global and local outlier detection for comprehensive coverage

### Reference-Free Detection
- **Self-Organizing Analysis**: Use test data's own structure to establish normality baselines
- **Statistical Robustness**: Combine multiple similarity measures for reliable anomaly scoring
- **Production Viability**: No dependency on external clean data sources

### Domain Adaptation
- **Column-Specific Optimization**: Tailored model architectures and hyperparameters per data type
- **Rule-Based Error Synthesis**: Domain-aware synthetic anomaly generation
- **Transfer Learning**: Leverage pre-trained semantic representations for faster convergence

## Performance & Scalability

### Computational Complexity
- **Training**: O(n²) for triplet generation, O(n) for model training
- **Inference**: O(n²) for similarity matrix computation, O(n) for scoring
- **Memory**: Linear scaling with dataset size and embedding dimensions

### Optimization Strategies
- **Batch Encoding**: Vectorized operations for efficient similarity computation
- **Early Stopping**: Prevent overfitting during training
- **Model Compression**: Use smaller models (384-dim vs 768-dim) when performance permits

## Architectural Extensions

### Near-Term Enhancements
- **Ensemble Methods**: Combine multiple model predictions for improved robustness
- **Online Learning**: Incremental model updates without full retraining
- **Uncertainty Estimation**: Bayesian approaches for confidence scoring

### Advanced Capabilities
- **Multi-Modal Detection**: Incorporate image/structured data alongside text
- **Causal Anomaly Detection**: Identify root causes of data quality issues
- **Real-Time Streaming**: Event-driven anomaly detection for live data pipelines

---

**Architecture Notes**: The system is designed for production ML pipelines requiring high recall anomaly detection without clean reference data dependencies. The triplet loss approach enables semantic understanding while the reference-free inference ensures practical deployment viability.
