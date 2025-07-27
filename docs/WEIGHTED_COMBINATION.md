# Weighted Combination Detection System

This document describes the weighted combination approach for anomaly detection, which provides an alternative to the priority-based system for combining results from multiple detection methods.

## Overview

The weighted combination system intelligently combines results from multiple anomaly detection methods (pattern-based, ML-based, and LLM-based) using field-specific weights calculated from performance metrics. This approach allows the system to dynamically prioritize the most effective detection method for each field.

## Key Features

- **Performance-based weighting**: Weights are calculated based on F1 scores from evaluation results
- **Field-specific optimization**: Each field can have different weights for different detection methods
- **JSON-based configuration**: Weights are stored in external JSON files for easy updates
- **Automatic fallback**: Falls back to equal weights when weights file is unavailable
- **Maintainable**: No hardcoded performance data in source code

## Architecture

### Components

1. **Weight Generation Tool** (`generate_detection_weights.py`)
2. **Weighted Detection Logic** (in `comprehensive_detector.py`)
3. **Demo Integration** (in `demo.py`)

## Usage

### 1. Generate Detection Weights

First, generate weights from evaluation results:

```bash
python3 generate_detection_weights.py \
    --input-file demo_results/demo_analysis_unified_report.json \
    --output-file detection_weights.json \
    --verbose
```

**Arguments:**
- `--input-file, -i`: Path to unified report JSON file with performance data (required)
- `--output-file, -o`: Output file for generated weights (default: `detection_weights.json`)
- `--baseline-weight, -b`: Baseline weight for untrained methods (default: 0.1)
- `--verbose, -v`: Print detailed weight information

### 2. Run Detection with Weighted Combination

Use the generated weights in detection:

```bash
python3 demo.py \
    --data-file data/esqualo_2022_fall_original.csv \
    --core-fields-only \
    --enable-validation \
    --enable-pattern \
    --enable-ml \
    --enable-llm \
    --use-weighted-combination \
    --weights-file detection_weights.json \
    --injection-intensity 0.15
```

**New Arguments:**
- `--use-weighted-combination`: Enable weighted combination instead of priority-based
- `--weights-file`: Path to JSON file containing detection weights

## Weight Calculation Algorithm

### 1. Performance Extraction
The system extracts precision and recall metrics for each detection method per field from evaluation results:

```python
precision = method_data.get('precision', 0.0)
recall = method_data.get('recall', 0.0)
f1_score = 2 * (precision * recall) / (precision + recall)
```

### 2. Weight Assignment
Weights are assigned based on F1 scores with a baseline for untrained methods:

```python
baseline_weight = 0.1
weight = max(f1_score, baseline_weight)
```

### 3. Normalization
Weights are normalized to sum to 1.0 for each field:

```python
total_weight = sum(all_weights)
normalized_weight = weight / total_weight
```

### 4. Detection Combination
During detection, method confidences are weighted and combined:

```python
weighted_confidence = method_confidence * method_weight
total_weighted_score = sum(all_weighted_confidences)
anomaly_detected = total_weighted_score >= threshold  # default: 0.3
```

## Weights File Format

The generated weights file contains:

```json
{
  "metadata": {
    "description": "Field-specific weights for anomaly detection methods",
    "source_file": "demo_results/demo_analysis_unified_report.json",
    "calculation_method": "F1-score based with baseline normalization",
    "baseline_weight": 0.1,
    "generated_by": "generate_detection_weights.py"
  },
  "weights": {
    "category": {
      "pattern_based": 0.83,
      "ml_based": 0.08,
      "llm_based": 0.08
    },
    "color_name": {
      "pattern_based": 0.83,
      "ml_based": 0.09,
      "llm_based": 0.09
    }
  },
  "weight_summary": {
    "category": {
      "dominant_method": "pattern_based",
      "dominant_weight": 0.83,
      "weights": {...}
    }
  },
  "performance_insights": {
    "category": [
      "pattern_based: Excellent (F1=1.000)",
      "ml_based: Not effective (F1=0.000)",
      "llm_based: Not trained/available"
    ]
  }
}
```

## Example Weight Scenarios

### High-Performing Method
```json
"category": {
  "pattern_based": 0.83,  // Excellent F1=1.0 performance
  "ml_based": 0.08,       // Baseline weight (untrained)
  "llm_based": 0.08       // Baseline weight (untrained)
}
```

### Poor-Performing Methods
```json
"material": {
  "pattern_based": 0.33,  // Poor F1=0.032, falls back to equal
  "ml_based": 0.33,       // Equal weight
  "llm_based": 0.33       // Equal weight
}
```

### Mixed Performance
```json
"color_name": {
  "pattern_based": 0.82,  // Very good F1=0.923
  "ml_based": 0.09,       // Baseline weight
  "llm_based": 0.09       // Baseline weight
}
```

## Validation Results Priority

**Important**: Validation (rule-based) results always have the highest priority and are applied regardless of the combination method. The weighted combination only applies to anomaly detection methods (pattern-based, ML-based, LLM-based).

## Implementation Details

### ComprehensiveFieldDetector Changes

1. **Constructor**: Added `use_weighted_combination` and `weights_file` parameters
2. **Weight Loading**: `_load_field_weights_from_file()` method loads weights from JSON
3. **Classification**: `classify_cells()` now routes to either priority-based or weighted classification
4. **Weighted Logic**: `_weighted_classify_cells()` implements the weighted combination algorithm

### Demo Integration

1. **CLI Arguments**: Added `--use-weighted-combination` and `--weights-file` flags
2. **Constructor**: Updated to accept weighted combination parameters
3. **Detector Creation**: Passes weighted combination settings to detector

## Performance Comparison

### Priority-Based Approach
- Uses fixed hierarchy: validation > pattern-based > ML-based > LLM-based
- First detection method to flag a cell wins
- Simple but may miss optimal combinations

### Weighted Combination Approach  
- Uses performance-based weights for each field
- Combines confidence scores from multiple methods
- More nuanced but requires evaluation data

## Monitoring and Updates

### When to Regenerate Weights

1. **New Models Trained**: After training new ML or LLM models
2. **Performance Changes**: When detection performance metrics change significantly
3. **New Fields Added**: When adding new fields to the system
4. **Rule Updates**: After updating validation rules that affect performance

### Weight Validation

The system automatically validates loaded weights:
- Ensures all required methods have weights
- Provides default weights for missing methods
- Falls back to equal weights if file is corrupted/missing

## Troubleshooting

### Common Issues

1. **Weights file not found**:
   ```
   ⚠️  Warning: Could not load weights from detection_weights.json: [Errno 2] No such file or directory
   ⚠️  Falling back to equal weights for all methods
   ```
   **Solution**: Generate weights file using `generate_detection_weights.py`

2. **Invalid JSON format**:
   ```
   ⚠️  Warning: Could not load weights from detection_weights.json: Expecting property name enclosed in double quotes
   ```
   **Solution**: Regenerate weights file or fix JSON syntax

3. **Missing performance data**:
   ```
   ❌ Error: No field performance data found in demo_results/report.json
   ```
   **Solution**: Run demo with ground truth data to generate performance metrics

### Debugging

Enable verbose output to see weight assignment:
```bash
python3 generate_detection_weights.py -i report.json -o weights.json --verbose
```

Monitor weight application during detection:
```
📊 Detection weights by field:
   category: pattern_based: 0.83, ml_based: 0.08, llm_based: 0.08
   color_name: pattern_based: 0.83, ml_based: 0.09, llm_based: 0.09
```

## Future Enhancements

1. **Dynamic Weight Updates**: Real-time weight adjustment based on streaming performance
2. **Cross-Validation**: Multiple evaluation runs for more robust weight calculation
3. **Method-Specific Thresholds**: Different anomaly thresholds per detection method
4. **Ensemble Learning**: Advanced combination techniques beyond weighted averaging