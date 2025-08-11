#!/bin/bash
# Script to extract individual model zips from model_zips/ directory
# Usage: ./extract_models.sh [options] [model_path1 model_path2 ...]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
# 
# Examples:
#   ./extract_models.sh                           # list available models
#   ./extract_models.sh ml/category/baseline      # extract specific model
#   ./extract_models.sh -f category               # list/extract category models only
#   ./extract_models.sh -v baseline               # list/extract baseline variations only
#   ./extract_models.sh -f category -v baseline   # list/extract category baseline only

set -e

# Check if we're in the data directory
if [ ! -f "extract_models.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

MODEL_ZIPS_DIR="model_zips"

# Parse command line arguments
FIELD_FILTER=""
VARIATION_FILTER=""
MODEL_PATHS=()

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
            echo "Usage: $0 [options] [model_path1 model_path2 ...]"
            echo "Options:"
            echo "  -f, --field FIELD          Filter by field name (e.g., category, material)"
            echo "  -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # list available models"
            echo "  $0 ml/category/baseline              # extract specific model"
            echo "  $0 -f category                        # list/extract category models only"
            echo "  $0 -v baseline                        # list/extract baseline variations only"
            echo "  $0 -f category -v baseline            # list/extract category baseline only"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            MODEL_PATHS+=("$1")
            shift
            ;;
    esac
done

# Function to check if model matches filters
matches_filters() {
    local model_path="$1"
    
    # Extract field and variation from model path
    local field_name=""
    local variant_name=""
    
    # Parse model path: ml/category/baseline or llm/category/baseline
    if [[ "$model_path" =~ ^(ml|llm)/([^/]+)/([^/]+)$ ]]; then
        field_name="${BASH_REMATCH[2]}"
        variant_name="${BASH_REMATCH[3]}"
    else
        return 1
    fi
    
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

# Function to extract model
extract_model() {
    local model_path="$1"
    
    # Handle new naming convention: {model_type}_{field_name}_{variant_name}.zip
    local model_type=""
    local field_name=""
    local variant_name=""
    
    # Parse model path: ml/field/variant or llm/field/variant
    if [[ "$model_path" =~ ^(ml|llm)/([^/]+)/([^/]+)$ ]]; then
        model_type="${BASH_REMATCH[1]}"
        field_name="${BASH_REMATCH[2]}"
        variant_name="${BASH_REMATCH[3]}"
    else
        echo "    ❌ Invalid model path format: $model_path (expected: ml/field/variant or llm/field/variant)"
        return 1
    fi
    
    # Try both naming conventions
    local zip_path="$MODEL_ZIPS_DIR/${model_path}.zip"
    local new_zip_path="$MODEL_ZIPS_DIR/$model_type/$field_name/${model_type}_${field_name}_${variant_name}.zip"
    
    if [ -f "$new_zip_path" ]; then
        zip_path="$new_zip_path"
    elif [ ! -f "$zip_path" ]; then
        echo "    ❌ Model not found: $zip_path or $new_zip_path"
        return 1
    fi
    
    echo "📦 Extracting $model_path..."
    unzip -o "$zip_path" >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "    ✅ Successfully extracted $model_path"
        
        # Show extraction path based on model path
        if [[ "$model_path" == ml/* ]]; then
            # ML model: ml/variation/field -> models/ml/variation/field
            variant_name=$(echo "$model_path" | cut -d'/' -f2)
            field_name=$(echo "$model_path" | cut -d'/' -f3)
            extracted_path="models/ml/$variant_name/$field_name"
        elif [[ "$model_path" == llm/* ]]; then
            # LLM model: llm/variation/field -> models/llm/variation/field
            variant_name=$(echo "$model_path" | cut -d'/' -f2)
            field_name=$(echo "$model_path" | cut -d'/' -f3)
            extracted_path="models/llm/$variant_name/$field_name"
        fi
        
        if [ -d "$extracted_path" ]; then
            size=$(du -sh "$extracted_path" | cut -f1)
            echo "       📁 Extracted to: $extracted_path ($size)"
        fi
        
        return 0
    else
        echo "    ❌ Failed to extract $model_path"
        return 1
    fi
}

# If no arguments provided, list available models
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    if [ ! -d "$MODEL_ZIPS_DIR" ] || [ "$(find $MODEL_ZIPS_DIR -name "*.zip" | wc -l)" -eq 0 ]; then
        echo "❌ No model zips found in $MODEL_ZIPS_DIR/"
        echo "💡 Run ./zip_models_individually.sh first to create individual model zips."
        exit 1
    fi
    
    echo "📁 Available models in $MODEL_ZIPS_DIR/:"
    
    # Apply filters if specified
    FILTER_INFO=""
    if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
        FILTER_INFO=" (filtered:"
        [ -n "$FIELD_FILTER" ] && FILTER_INFO="$FILTER_INFO field=$FIELD_FILTER"
        [ -n "$VARIATION_FILTER" ] && FILTER_INFO="$FILTER_INFO variation=$VARIATION_FILTER"
        FILTER_INFO="$FILTER_INFO)"
    fi
    echo "$FILTER_INFO"
    echo ""
    
    # Collect and filter models
    ML_MODELS=()
    LLM_MODELS=()
    
    # Find all zip files and convert to model paths
    while IFS= read -r -d '' zip_file; do
        # Convert zip path to model path
        # Handle both old format: model_zips/ml/category/baseline.zip -> ml/category/baseline
        # and new format: model_zips/ml/category/ml_category_baseline.zip -> ml/category/baseline
        local relative_path=$(echo "$zip_file" | sed "s|$MODEL_ZIPS_DIR/||")
        local model_path=""
        
        # Check if it's the new naming convention: {model_type}_{field_name}_{variant_name}.zip
        if [[ "$relative_path" =~ ^(ml|llm)/([^/]+)/([^/]+)_([^/]+)_([^/]+)\.zip$ ]]; then
            local model_type="${BASH_REMATCH[1]}"
            local field_name="${BASH_REMATCH[2]}"
            local variant_name="${BASH_REMATCH[5]}"  # Last part after the last underscore
            model_path="$model_type/$field_name/$variant_name"
        else
            # Old naming convention: just remove .zip
            model_path=$(echo "$relative_path" | sed 's|\.zip$||')
        fi
        
        if matches_filters "$model_path"; then
            if [[ "$model_path" == ml/* ]]; then
                ML_MODELS+=("$model_path")
            elif [[ "$model_path" == llm/* ]]; then
                LLM_MODELS+=("$model_path")
            fi
        fi
    done < <(find "$MODEL_ZIPS_DIR" -name "*.zip" -print0)
    
    echo "🔍 ML Models (${#ML_MODELS[@]} found):"
    if [ ${#ML_MODELS[@]} -gt 0 ]; then
        printf '%s\n' "${ML_MODELS[@]}" | sort | while read model; do
            # Find the actual zip file for this model
            local model_type=$(echo "$model" | cut -d'/' -f1)
            local field_name=$(echo "$model" | cut -d'/' -f2)
            local variant_name=$(echo "$model" | cut -d'/' -f3)
            local zip_path="$MODEL_ZIPS_DIR/$model_type/$field_name/${model_type}_${field_name}_${variant_name}.zip"
            
            # Fall back to old naming convention if new one doesn't exist
            if [ ! -f "$zip_path" ]; then
                zip_path="$MODEL_ZIPS_DIR/${model}.zip"
            fi
            
            if [ -f "$zip_path" ]; then
                size=$(du -h "$zip_path" | cut -f1)
                echo "  📦 $model ($size)"
            else
                echo "  📦 $model (file not found)"
            fi
        done
    else
        echo "  (none matching filters)"
    fi
    
    echo ""
    echo "🤖 LLM Models (${#LLM_MODELS[@]} found):"
    if [ ${#LLM_MODELS[@]} -gt 0 ]; then
        printf '%s\n' "${LLM_MODELS[@]}" | sort | while read model; do
            # Find the actual zip file for this model
            local model_type=$(echo "$model" | cut -d'/' -f1)
            local field_name=$(echo "$model" | cut -d'/' -f2)
            local variant_name=$(echo "$model" | cut -d'/' -f3)
            local zip_path="$MODEL_ZIPS_DIR/$model_type/$field_name/${model_type}_${field_name}_${variant_name}.zip"
            
            # Fall back to old naming convention if new one doesn't exist
            if [ ! -f "$zip_path" ]; then
                zip_path="$MODEL_ZIPS_DIR/${model}.zip"
            fi
            
            if [ -f "$zip_path" ]; then
                size=$(du -h "$zip_path" | cut -f1)
                echo "  📦 $model ($size)"
            else
                echo "  📦 $model (file not found)"
            fi
        done
    else
        echo "  (none matching filters)"
    fi
    
    echo ""
    echo "💡 To extract models, run:"
    echo "   $0 <model_path>                    # extract specific model"
    echo "   $0 -f category                    # extract all category models"
    echo "   $0 -v baseline                    # extract all baseline variations"
    echo "   $0 -f category -v baseline        # extract category baseline models"
    exit 0
fi

# Extract specified models
echo "📦 Extracting models..."
EXTRACTED_COUNT=0

# If filters are specified, extract all matching models
if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
    echo "🔍 Applying filters:"
    [ -n "$FIELD_FILTER" ] && echo "  - Field: $FIELD_FILTER"
    [ -n "$VARIATION_FILTER" ] && echo "  - Variation: $VARIATION_FILTER"
    echo ""
    
    # Collect all matching models
    MATCHING_MODELS=()
    
    # Find all zip files and convert to model paths
    while IFS= read -r -d '' zip_file; do
        # Convert zip path to model path
        # Handle both old format: model_zips/ml/category/baseline.zip -> ml/category/baseline
        # and new format: model_zips/ml/category/ml_category_baseline.zip -> ml/category/baseline
        local relative_path=$(echo "$zip_file" | sed "s|$MODEL_ZIPS_DIR/||")
        local model_path=""
        
        # Check if it's the new naming convention: {model_type}_{field_name}_{variant_name}.zip
        if [[ "$relative_path" =~ ^(ml|llm)/([^/]+)/([^/]+)_([^/]+)_([^/]+)\.zip$ ]]; then
            local model_type="${BASH_REMATCH[1]}"
            local field_name="${BASH_REMATCH[2]}"
            local variant_name="${BASH_REMATCH[5]}"  # Last part after the last underscore
            model_path="$model_type/$field_name/$variant_name"
        else
            # Old naming convention: just remove .zip
            model_path=$(echo "$relative_path" | sed 's|\.zip$||')
        fi
        
        if matches_filters "$model_path"; then
            MATCHING_MODELS+=("$model_path")
        fi
    done < <(find "$MODEL_ZIPS_DIR" -name "*.zip" -print0)
    
    if [ ${#MATCHING_MODELS[@]} -eq 0 ]; then
        echo "❌ Error: No models found matching the specified filters." >&2
        echo "💡 Available models:" >&2
        find "$MODEL_ZIPS_DIR" -name "*.zip" | sed "s|$MODEL_ZIPS_DIR/||" | sed 's|\.zip$||' | head -10
        if [ "$(find "$MODEL_ZIPS_DIR" -name "*.zip" | wc -l)" -gt 10 ]; then
            echo "  ... and $(($(find "$MODEL_ZIPS_DIR" -name "*.zip" | wc -l) - 10)) more"
        fi
        exit 1
    fi
    
    echo "📊 Found ${#MATCHING_MODELS[@]} matching models to extract:"
    for model in "${MATCHING_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    
    # Extract matching models
    for model_path in "${MATCHING_MODELS[@]}"; do
        if extract_model "$model_path"; then
            EXTRACTED_COUNT=$((EXTRACTED_COUNT + 1))
        fi
    done
else
    # Extract specific models provided as arguments
    for model_path in "${MODEL_PATHS[@]}"; do
        if extract_model "$model_path"; then
            EXTRACTED_COUNT=$((EXTRACTED_COUNT + 1))
        else
            echo "    💡 Available models:"
            find "$MODEL_ZIPS_DIR" -name "*.zip" | sed "s|$MODEL_ZIPS_DIR/||" | sed 's|\.zip$||' | head -5
            if [ "$(find "$MODEL_ZIPS_DIR" -name "*.zip" | wc -l)" -gt 5 ]; then
                echo "      ... and $(($(find "$MODEL_ZIPS_DIR" -name "*.zip" | wc -l) - 5)) more"
            fi
        fi
    done
fi

echo ""
echo "🎯 Extraction completed!"
echo "📊 Total models extracted: $EXTRACTED_COUNT"
echo "📁 Models are now available in the models/ directory"
