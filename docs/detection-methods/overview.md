# Detection Methods Overview

The Data Quality Detection System employs four complementary detection methods, each designed to identify different types of data quality issues with varying levels of confidence. This document provides an overview of all detection methods and guidance on when to use each one.

## Detection Method Comparison

```mermaid
graph LR
    subgraph "Detection Methods"
        V[Validation<br/>Rule-Based]
        P[Pattern-Based<br/>Anomaly Detection]
        M[ML-Based<br/>Semantic Detection]
        L[LLM-Based<br/>Context-Aware]
    end
    
    subgraph "Characteristics"
        V --> V1[100% Confidence<br/>No Training<br/>Fast]
        P --> P1[70-80% Confidence<br/>No Training<br/>Fast]
        M --> M1[Configurable Confidence<br/>Requires Training<br/>Medium Speed]
        L --> L1[Configurable Confidence<br/>Requires Training<br/>Slower]
    end
```

## Method Summary

| Method | Confidence | Training Required | Speed | Best Use Cases |
|--------|------------|------------------|--------|----------------|
| **Validation** | 100% | No | Very Fast | Format errors, business rules, required fields |
| **Pattern-Based** | 70-80% | No | Fast | Known patterns, outliers, statistical anomalies |
| **ML-Based** | Configurable | Yes | Medium | Semantic similarity, contextual errors |
| **LLM-Based** | Configurable | Yes | Slower | Complex logic, natural language understanding |

## Detection Flow

```mermaid
flowchart TD
    A[Input Data] --> B{Field Type}
    B --> C[Load Configuration]
    
    C --> D[Validation]
    C --> E[Pattern Detection]
    C --> F[ML Detection]
    C --> G[LLM Detection]
    
    D --> H{Errors Found?}
    E --> I{Anomalies Found?}
    F --> J{Anomalies Found?}
    G --> K{Anomalies Found?}
    
    H & I & J & K --> L[Combine Results]
    
    L --> M{Combination Strategy}
    M --> N[Priority-Based]
    M --> O[Weighted Average]
    
    N & O --> P[Final Decision]
    P --> Q[Generate Report]
```

## Method Selection Guide

### Use Validation When:
- ✅ You need 100% confidence in error detection
- ✅ Checking format compliance (emails, phone numbers, etc.)
- ✅ Enforcing business rules (required fields, value ranges)
- ✅ Speed is critical
- ❌ You need to detect subtle or contextual errors

### Use Pattern-Based Detection When:
- ✅ You have known good patterns or values
- ✅ Detecting statistical outliers
- ✅ No training data available
- ✅ Need fast detection with reasonable accuracy
- ❌ Dealing with complex semantic relationships

### Use ML-Based Detection When:
- ✅ You have training data available
- ✅ Detecting semantic anomalies
- ✅ Context matters for detection
- ✅ GPU acceleration is available
- ❌ You need explainable decisions

### Use LLM-Based Detection When:
- ✅ Complex natural language understanding required
- ✅ Context spans multiple fields
- ✅ Need sophisticated reasoning
- ✅ Have computational resources
- ❌ Speed is critical

## Confidence Levels

Each method operates at different confidence levels:

```mermaid
graph TD
    subgraph "Confidence Spectrum"
        A[100% - Validation<br/>Definite Errors]
        B[70-80% - Pattern<br/>Likely Anomalies]
        C[60-75% - ML<br/>Semantic Anomalies]
        D[50-70% - LLM<br/>Context Anomalies]
    end
    
    A --> E[High Precision<br/>Low Recall]
    D --> F[Lower Precision<br/>Higher Recall]
```

## Combination Strategies

The system supports two strategies for combining detection results:

### 1. Priority-Based Combination
Methods are prioritized by confidence level:
1. Validation (highest priority)
2. Pattern-Based
3. ML-Based
4. LLM-Based

```python
if validation_error:
    return validation_error
elif pattern_anomaly:
    return pattern_anomaly
elif ml_anomaly:
    return ml_anomaly
else:
    return llm_anomaly
```

### 2. Weighted Average Combination
Results are combined using learned weights:

```python
final_score = (
    validation_weight * validation_score +
    pattern_weight * pattern_score +
    ml_weight * ml_score +
    llm_weight * llm_score
)
```

## Performance Considerations

```mermaid
graph LR
    subgraph "Processing Time"
        A[Validation<br/>~1ms/record]
        B[Pattern<br/>~5ms/record]
        C[ML<br/>~20ms/record]
        D[LLM<br/>~100ms/record]
    end
    
    subgraph "Resource Usage"
        A --> A1[CPU Only<br/>Low Memory]
        B --> B1[CPU Only<br/>Low Memory]
        C --> C1[GPU Optional<br/>Medium Memory]
        D --> D1[GPU Recommended<br/>High Memory]
    end
```

## Field-Specific Configurations

Different fields may benefit from different detection methods:

| Field Type | Recommended Methods | Rationale |
|------------|-------------------|-----------|
| **Material** | Validation + ML | Semantic understanding needed |
| **Color** | Pattern + ML | Known patterns with variations |
| **Size** | Validation + Pattern | Strict formats with known values |
| **Description** | ML + LLM | Natural language processing |
| **Price** | Validation + Pattern | Numerical rules and outliers |

## Enabling/Disabling Methods

Control which methods run via command line:

```bash
# Run all methods (default behavior when no methods are explicitly enabled)
python main.py single-demo \
    --data-file data.csv

# Run specific methods
python main.py single-demo \
    --data-file data.csv \
    --enable-validation \
    --enable-pattern

# Adjust thresholds
python main.py single-demo \
    --data-file data.csv \
    --validation-threshold 0.0 \
    --anomaly-threshold 0.8 \
    --ml-threshold 0.75
```

## Method Dependencies

```mermaid
graph TD
    A[Base Dependencies]
    A --> B[pandas, numpy]
    
    B --> C[Validation]
    B --> D[Pattern-Based]
    
    B --> E[scikit-learn]
    E --> F[ML-Based]
    
    F --> G[sentence-transformers]
    G --> H[ML Models]
    
    F --> I[transformers]
    I --> J[LLM-Based]
    
    K[Optional: CUDA/GPU]
    K --> H
    K --> J
```

## Choosing the Right Mix

For different use cases, consider these combinations:

### High-Speed Screening
- ✅ Validation + Pattern-Based
- ❌ ML + LLM
- **Use when**: Processing large volumes quickly

### Comprehensive Analysis
- ✅ All methods enabled
- **Use when**: Thorough analysis needed

### Production Monitoring
- ✅ Validation + Weighted ML
- **Use when**: Balance of speed and accuracy

### Development/Testing
- ✅ All methods with error injection
- **Use when**: Evaluating system performance

## Next Steps

- Explore the [API Reference](../api/interfaces.md)
- Learn about [Configuration](../configuration/brand-config.md)
- Read about [Adding New Fields](../development/new-fields.md)