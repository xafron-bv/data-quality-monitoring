#!/bin/bash
# Script to zip each model variant individually for GitHub releases
# This creates separate zip files for each model variant to stay under GitHub's 2GB limit
# Uses parallel processing with all available CPU cores
# 
# Usage: ./zip_models_individually.sh [options]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
#
# Examples:
#   ./zip_models_individually.sh                    # All models
#   ./zip_models_individually.sh -f category        # Only category models
#   ./zip_models_individually.sh -v baseline        # Only baseline variations
#   ./zip_models_individually.sh -f category -v baseline  # Only category baseline

set -e

# Check if we're in the data directory
if [ ! -f "zip_models_individually.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

MODELS_DIR="models"
OUTPUT_DIR="model_zips"

# Parse command line arguments
FIELD_FILTER=""
VARIATION_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--field)
            FIELD_FILTER="$2"
            shift 2
            ;;
        -v|--variation)
            VARIATION_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -f, --field FIELD          Filter by field name (e.g., category, material)"
            echo "  -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # All models"
            echo "  $0 -f category                       # Only category models"
            echo "  $0 -v baseline                       # Only baseline variations"
            echo "  $0 -f category -v baseline           # Only category baseline"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get number of CPU cores (use 8 as specified)
CPU_CORES=8

# Display filters if specified
FILTER_INFO=""
if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
    FILTER_INFO=" with filters:"
    [ -n "$FIELD_FILTER" ] && FILTER_INFO="$FILTER_INFO field=$FIELD_FILTER"
    [ -n "$VARIATION_FILTER" ] && FILTER_INFO="$FILTER_INFO variation=$VARIATION_FILTER"
fi

echo "🚀 Starting individual model zipping with $CPU_CORES parallel processes$FILTER_INFO..."

# Create output directory structure
mkdir -p "$OUTPUT_DIR/ml"
mkdir -p "$OUTPUT_DIR/llm"

# Function to get directory size in human readable format
get_dir_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" | cut -f1
    else
        echo "0B"
    fi
}

# Function to zip a model variant (will be called in parallel)
zip_model_variant() {
    local model_path="$1"
    local model_type="$2"
    local field_name="$3"
    local variant_name="$4"
    
    # Create the directory structure if it doesn't exist
    local output_subdir="$OUTPUT_DIR/$model_type/$field_name"
    mkdir -p "$output_subdir"
    
    # Create unique filename: {model_type}_{field_name}_{variant_name}.zip
    local zip_name="${model_type}_${field_name}_${variant_name}.zip"
    local zip_path="$output_subdir/$zip_name"
    
    echo "  📦 Zipping: $model_type/$field_name/$variant_name (PID: $$)"
    
    # Remove existing zip if it exists
    [ -f "$zip_path" ] && rm -f "$zip_path"
    
    # Create zip file from the models directory
    cd "$MODELS_DIR"
    if [ -d "$model_path" ]; then
        zip -r "../$zip_path" "$model_path" -q
        cd ..
        
        local size=$(get_dir_size "$MODELS_DIR/$model_path")
        local zip_size=$(du -h "$zip_path" | cut -f1)
        
        echo "    ✅ Created: $model_type/$field_name/$zip_name (${size} -> ${zip_size})"
    else
        cd ..
        echo "    ❌ Directory not found: $model_path"
    fi
}

# Function to check if model matches filters
matches_filters() {
    local field_name="$1"
    local variant_name="$2"
    
    # Check field filter
    if [ -n "$FIELD_FILTER" ] && [[ "$field_name" != *"$FIELD_FILTER"* ]]; then
        return 1
    fi
    
    # Check variation filter
    if [ -n "$VARIATION_FILTER" ] && [[ "$variant_name" != *"$VARIATION_FILTER"* ]]; then
        return 1
    fi
    
    return 0
}

# Collect all model variants to process
echo "📁 Collecting model variants..."
MODEL_VARIANTS=()

# Collect ML models
if [ -d "$MODELS_DIR/ml/trained" ]; then
    for field_dir in "$MODELS_DIR/ml/trained"/*; do
        if [ -d "$field_dir" ]; then
            field_name=$(basename "$field_dir")
            for variant_dir in "$field_dir"/*; do
                if [ -d "$variant_dir" ]; then
                    variant_name=$(basename "$variant_dir")
                    
                    # Check if this model matches our filters
                    if matches_filters "$field_name" "$variant_name"; then
                        model_path="ml/trained/$field_name/$variant_name"
                        MODEL_VARIANTS+=("$model_path|ml|$field_name|$variant_name")
                    fi
                fi
            done
        fi
    done
fi

# Collect LLM models
if [ -d "$MODELS_DIR/llm" ]; then
    for model_dir in "$MODELS_DIR/llm"/*_model; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            # Extract field name from model name (remove _model suffix)
            field_name=$(echo "$model_name" | sed 's/_model$//')
            
            for variant_dir in "$model_dir"/*; do
                if [ -d "$variant_dir" ]; then
                    variant_name=$(basename "$variant_dir")
                    
                    # Check if this model matches our filters
                    if matches_filters "$field_name" "$variant_name"; then
                        model_path="llm/$model_name/$variant_name"
                        MODEL_VARIANTS+=("$model_path|llm|$field_name|$variant_name")
                    fi
                fi
            done
        fi
    done
fi

echo "📊 Found ${#MODEL_VARIANTS[@]} model variants to process"

if [ ${#MODEL_VARIANTS[@]} -eq 0 ]; then
    echo "❌ Error: No models found matching the specified filters." >&2
    if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
        echo "💡 Available fields and variations:" >&2
        echo "  Fields: $(find "$MODELS_DIR" -type d -name "*" | grep -E "(ml/trained|llm/.*_model)" | sed 's/.*\///' | sort -u | tr '\n' ' ')" >&2
        echo "  Variations: $(find "$MODELS_DIR" -type d -mindepth 2 | sed 's/.*\///' | sort -u | tr '\n' ' ')" >&2
    fi
    exit 1
fi

# Process models in parallel using background processes
if [ ${#MODEL_VARIANTS[@]} -gt 0 ]; then
    echo "🔄 Starting parallel processing with $CPU_CORES cores..."
    
    # Array to store background process PIDs
    PIDS=()
    CURRENT_JOBS=0
    
    for variant_info in "${MODEL_VARIANTS[@]}"; do
        IFS='|' read -r model_path model_type field_name variant_name <<< "$variant_info"
        
        # If we've reached the max number of parallel jobs, wait for one to finish
        while [ $CURRENT_JOBS -ge $CPU_CORES ]; do
            # Check if any background process has finished
            for i in "${!PIDS[@]}"; do
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    # Process has finished, remove it from the array
                    unset PIDS[$i]
                    CURRENT_JOBS=$((CURRENT_JOBS - 1))
                fi
            done
            
            # If no processes finished, wait a bit
            if [ $CURRENT_JOBS -ge $CPU_CORES ]; then
                sleep 1
            fi
        done
        
        # Start a new background process
        (
            zip_model_variant "$model_path" "$model_type" "$field_name" "$variant_name"
        ) &
        
        PIDS+=($!)
        CURRENT_JOBS=$((CURRENT_JOBS + 1))
        
        echo "  🚀 Started job $CURRENT_JOBS/$CPU_CORES for $model_type/$field_name/$variant_name (PID: $!)"
    done
    
    # Wait for all remaining background processes to finish
    echo "⏳ Waiting for all jobs to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    echo "✅ All parallel jobs completed!"
else
    echo "⚠️  No model variants found to process"
fi

# Create a summary file
echo "📊 Creating summary..."
{
    echo "# Model Variants Summary"
    echo "Generated on: $(date)"
    echo "Processed with $CPU_CORES parallel cores"
    if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
        echo "Filters applied:"
        [ -n "$FIELD_FILTER" ] && echo "- Field: $FIELD_FILTER"
        [ -n "$VARIATION_FILTER" ] && echo "- Variation: $VARIATION_FILTER"
    fi
    echo ""
    echo "## Directory Structure"
    echo "\`\`\`"
    echo "$OUTPUT_DIR/"
    echo "├── ml/"
    echo "│   ├── category/"
    echo "│   │   └── baseline.zip"
    echo "│   ├── material/"
    echo "│   │   └── baseline.zip"
    echo "│   └── ..."
    echo "└── llm/"
    echo "    ├── category/"
    echo "    │   ├── baseline.zip"
    echo "    │   └── v1.zip"
    echo "    ├── material/"
    echo "    │   └── baseline.zip"
    echo "    └── ..."
    echo "\`\`\`"
    echo ""
    echo "## ML Models"
    if [ -d "$OUTPUT_DIR/ml" ]; then
        find "$OUTPUT_DIR/ml" -name "*.zip" | sort | while read file; do
            size=$(du -h "$file" | cut -f1)
            relative_path=$(echo "$file" | sed "s|$OUTPUT_DIR/||")
            echo "- $relative_path ($size)"
        done
    fi
    echo ""
    echo "## LLM Models"
    if [ -d "$OUTPUT_DIR/llm" ]; then
        find "$OUTPUT_DIR/llm" -name "*.zip" | sort | while read file; do
            size=$(du -h "$file" | cut -f1)
            relative_path=$(echo "$file" | sed "s|$OUTPUT_DIR/||")
            echo "- $relative_path ($size)"
        done
    fi
} > "$OUTPUT_DIR/README.md"

# Show final summary
echo ""
echo "🎯 Individual model zipping completed!"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Total zip files created: $(find "$OUTPUT_DIR" -name "*.zip" | wc -l)"
echo "📄 Summary: $OUTPUT_DIR/README.md"

# Show directory structure
echo ""
echo "📁 Directory structure created:"
echo "  $OUTPUT_DIR/"
echo "  ├── ml/"
echo "  │   └── [field]/[variation].zip"
echo "  └── llm/"
echo "      └── [field]/[variation].zip"

# Show largest files
echo ""
echo "📏 Largest zip files:"
find "$OUTPUT_DIR" -name "*.zip" -exec du -h {} \; | sort -h | tail -5 | while read size file; do
    relative_path=$(echo "$file" | sed "s|$OUTPUT_DIR/||")
    echo "  $relative_path ($size)"
done
