# 🧪 Demo 1: Validation + Pattern-Based Detection Only

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
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
python single_sample_multi_field_demo.py \
  --brand your_brand \
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
python single_sample_multi_field_demo.py \
  --brand your_brand \
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
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_ml_only \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --ml-threshold 0.8 \
  --core-fields-only \
  --enable-ml
```

# 🧪 Demo 5: Validation only

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_validation_only \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
```

# 🧪 Demo 6: LLM-Based Detection Only

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_llm_only \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --llm-threshold 0.6 \
  --core-fields-only \
  --enable-llm
```

# 🧪 Demo 7: Validation + LLM-Based Detection

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_validation_llm \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
  --llm-threshold 0.6 \
  --core-fields-only \
  --enable-validation \
  --enable-llm
```

# 🧪 Demo 8: All Detection Methods (Validation + Pattern + ML + LLM)

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_all_methods_with_llm \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --validation-threshold 0.0 \
  --anomaly-threshold 0.7 \
  --ml-threshold 0.8 \
  --llm-threshold 0.6 \
  --core-fields-only \
  --enable-validation \
  --enable-pattern \
  --enable-ml \
  --enable-llm
```

# 🧪 Demo 9: LLM-Based Detection with Few-Shot Examples

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_llm_few_shot \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --llm-threshold 0.6 \
  --core-fields-only \
  --enable-llm \
  --llm-few-shot-examples
```

# 🧪 Demo 10: LLM-Based Detection with Dynamic Context

```bash
python single_sample_multi_field_demo.py \
  --brand your_brand \
  --output-dir demo_comparison_llm_dynamic \
  --injection-intensity 0.15 \
  --max-issues-per-row 2 \
  --llm-threshold 0.6 \
  --core-fields-only \
  --enable-llm \
  --llm-temporal-column timestamp \
  --llm-context-columns category,brand,season
```