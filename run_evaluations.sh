#!/bin/zsh

# run_evaluations.sh
# Script to run multiple evaluate.py commands for different column and validator combinations

# Default parameters
DATA_FILE="data/esqualo_2022_fall_original.csv"
MAX_ERRORS=5
NUM_SAMPLES=100
OUTPUT_DIR="evaluation_results"
IGNORE_FP="--ignore-fp"

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
  
  # Run the evaluation
  python evaluate.py \
    --validator="$validator" \
    --column="$column" \
    --max-errors=$MAX_ERRORS \
    --num-samples=$NUM_SAMPLES \
    --output-dir="$eval_dir" \
    $IGNORE_FP \
    "$DATA_FILE"
  
  echo "Evaluation complete for $validator:$column"
  echo "----------------------------------------"
  echo ""
done

echo "All evaluations completed!"
