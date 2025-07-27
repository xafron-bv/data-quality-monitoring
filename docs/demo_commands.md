# Demo Commands

## Basic Usage Examples

### 1. Validation + Pattern Detection
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_validation_pattern \
  --injection-intensity 0.15 \
  --enable-validation \
  --enable-pattern
```

### 2. All Detection Methods
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_all_methods \
  --injection-intensity 0.15 \
  --enable-validation \
  --enable-pattern \
  --enable-ml \
  --enable-llm
```

### 3. Weighted Combination
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_weighted \
  --enable-validation \
  --enable-pattern \
  --enable-ml \
  --use-weighted-combination \
  --weights-file detection_weights.json
```

### 4. Core Fields Only
```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --core-fields-only \
  --enable-validation \
  --enable-pattern
```