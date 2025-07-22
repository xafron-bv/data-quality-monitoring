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

# Define the column and validator pairs
# Format: "validator_name:column_name"
EVALUATIONS=(
  "category:article_structure_name_2"
  "season:season"
  "color_name:colour_name"
  "care_instructions:Care Instructions"
)

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each evaluation
for eval in "${EVALUATIONS[@]}"; do
  # Split the string into validator and column
  validator=$(echo $eval | cut -d':' -f1)
  column=$(echo $eval | cut -d':' -f2)
  
  # Create a directory for this specific evaluation
  eval_dir="${OUTPUT_DIR}/${validator}_${column// /_}"
  mkdir -p "$eval_dir"
  
  echo "----------------------------------------"
  echo "Running evaluation for:"
  echo "Validator: $validator"
  echo "Column: $column"
  echo "Output directory: $eval_dir"
  
  # Run the evaluation with all three detection methods
  python evaluate.py \
    --validator="$validator" \
    --column="$column" \
    --max-errors=$MAX_ERRORS \
    --num-samples=$NUM_SAMPLES \
    --output-dir="$eval_dir" \
    --ml-detector \
    --run="all" \
    $IGNORE_FP \
    $DEBUG_FLAG \
    "$DATA_FILE"
  
  echo "Evaluation complete for $validator:$column"
  echo "----------------------------------------"
  echo ""
done

echo "All evaluations completed!"
