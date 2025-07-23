#!/bin/zsh

# run_evaluations.sh
# Script to run comprehensive evaluations using all three detection approaches:
# 1. Validation (business rules)
# 2. Anomaly Detection (pattern-based)  
# 3. ML Anomaly Detection (sentence transformers)

# Default parameters
DATA_FILE="data/esqualo_2022_fall_original.csv"
MAX_ERRORS=5
NUM_SAMPLES=100
OUTPUT_DIR="evaluation_results"
IGNORE_FP="--ignore-fp"
DEBUG_FLAG=""  # Set to "--debug" to enable debug logging

# Dynamically fetch all fields from the centralized field mapper
for field in $(python -c "from common_interfaces import FieldMapper; fm = FieldMapper.from_default_mapping(); print(' '.join(fm.get_available_fields()))"); do
  # Create a directory for this specific evaluation
  eval_dir="${OUTPUT_DIR}/${field// /_}"
  mkdir -p "$eval_dir"

  echo "----------------------------------------"
  echo "Running evaluation for:"
  echo "Field: $field"
  echo "Output directory: $eval_dir"

  # Run the evaluation with all three detection methods
  python evaluate.py \
    --validator="$field" \
    --field="$field" \
    --max-errors=$MAX_ERRORS \
    --num-samples=$NUM_SAMPLES \
    --output-dir="$eval_dir" \
    --ml-detector \
    --run="all" \
    $IGNORE_FP \
    $DEBUG_FLAG \
    "$DATA_FILE"

  echo "Evaluation complete for $field"
  echo "----------------------------------------"
  echo ""
done

echo "All evaluations completed!"
