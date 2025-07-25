# 🧪 Demo 1: Validation + Pattern-Based Detection Only

```bash
python demo.py \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir demo_comparison_validation_pattern \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
  --anomaly-threshold 0.7 \
  --ml-threshold 1.0 \
  --core-fields-only \
  --enable-validation \
  --enable-pattern
```

# 🧪 Demo 2: Validation + ML-Based Detection Only

```bash
python demo.py \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir demo_comparison_validation_ml \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
  --anomaly-threshold 1.0 \
  --ml-threshold 0.8 \
  --core-fields-only \
  --enable-validation \
  --enable-ml
```

# 🧪 Demo 3: Validation + Pattern-Based + ML-Based Detection

```bash
python demo.py \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir demo_comparison_all_methods \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
  --anomaly-threshold 0.7 \
  --ml-threshold 0.8 \
  --core-fields-only \
  --enable-validation \
  --enable-pattern \
  --enable-ml
```

# 🧪 Demo 4: ML-Based Detection Only

```bash
python demo.py \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir demo_comparison_ml_only \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --ml-threshold 0.8 \
  --core-fields-only \
  --enable-ml
```

# 🧪 Demo 5: Validation only

```bash
python demo.py \
  --data-file data/esqualo_2022_fall.csv \
  --output-dir demo_comparison_validation_only \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
```