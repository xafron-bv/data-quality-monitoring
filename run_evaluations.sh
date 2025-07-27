#!/bin/zsh

# run_evaluations.sh
# Script to run comprehensive evaluations using all three detection approaches:
# 1. Validation (business rules)
# 2. Anomaly Detection (pattern-based)  
# 3. ML Anomaly Detection (sentence transformers)

# Check if brand is provided as first argument
if [ $# -lt 1 ]; then
    echo "Error: Brand name required"
    echo "Usage: $0 <brand_name> [data_file]"
    exit 1
fi

BRAND=$1
echo "Using brand: $BRAND"

# Default parameters
DATA_FILE="data/your_training_data.csv"
if [ $# -ge 2 ]; then
    DATA_FILE=$2
fi

MAX_ERRORS=5
NUM_SAMPLES=100
OUTPUT_DIR="evaluation_results"
IGNORE_FP="--ignore-fp"
DEBUG_FLAG=""  # Set to "--debug" to enable debug logging

# Dynamically fetch all fields from the brand configuration
for field in $(python -c "import sys; sys.path.append('.'); from static_brand_config import get_field_mappings; print(' '.join(get_field_mappings().keys()))"); do
  # Create a directory for this specific evaluation
  eval_dir="${OUTPUT_DIR}/${field// /_}"
  mkdir -p "$eval_dir"

  echo "----------------------------------------"
  echo "Running evaluation for:"
  echo "Field: $field"
  echo "Output directory: $eval_dir"

  # Run the evaluation with all three detection methods
  python evaluate_main.py \
    --brand="$BRAND" \
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
