#!/bin/zsh

# run_anomaly_detection.sh
# Script to run anomaly detection for different columns

# Default parameters
DATA_FILE="data/esqualo_2022_fall_original.csv"
NUM_SAMPLES=32
OUTPUT_DIR="evaluation_results/anomaly_detection"

# Define the column and anomaly detector pairs
# Format: "detector_name:column_name"
ANOMALY_DETECTIONS=(
  "material:material"
  "color_name:colour_name"
)

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each anomaly detection
for detection in "${ANOMALY_DETECTIONS[@]}"; do
  # Split the string into detector and column
  detector=$(echo $detection | cut -d':' -f1)
  column=$(echo $detection | cut -d':' -f2)
  
  # Create a directory for this specific evaluation
  detector_dir="${OUTPUT_DIR}/${detector}_${column// /_}"
  mkdir -p "$detector_dir"
  
  echo "----------------------------------------"
  echo "Running anomaly detection for:"
  echo "Detector: $detector"
  echo "Column: $column"
  echo "Output directory: $detector_dir"
  
  # Run the anomaly detection
  python evaluate.py \
    --anomaly-detector="$detector" \
    --column="$column" \
    --num-samples=$NUM_SAMPLES \
    --output-dir="$detector_dir" \
    --run="anomaly" \
    "$DATA_FILE"
  
  echo "Anomaly detection complete for $detector on $column"
  echo "----------------------------------------"
done

echo "All anomaly detection tasks completed!"
