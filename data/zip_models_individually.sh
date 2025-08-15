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

# Maximum file size in bytes (2GB - 100MB for safety margin)
MAX_FILE_SIZE=$((2000 * 1024 * 1024))  # 2000 MB

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

# Function to split a large zip file into parts
split_zip_file() {
    local zip_path="$1"
    local base_name="${zip_path%.zip}"
    local zip_size=$(stat -c%s "$zip_path" 2>/dev/null || stat -f%z "$zip_path" 2>/dev/null)
    
    if [ -z "$zip_size" ] || [ "$zip_size" -le "$MAX_FILE_SIZE" ]; then
        # File is small enough, no need to split
        return 0
    fi
    
    echo "    🔀 Splitting large zip file ($(du -h "$zip_path" | cut -f1)) into parts..."
    
    # Split the file into parts
    # Use 1900MB parts to ensure each part is well under 2GB
    split -b 1900M "$zip_path" "${base_name}.part"
    
    # Rename parts to have .zip extension for better GitHub release handling
    local part_num=1
    for part in "${base_name}".part*; do
        if [ -f "$part" ]; then
            mv "$part" "${base_name}.part${part_num}.zip"
            echo "      📦 Created part ${part_num}: $(du -h "${base_name}.part${part_num}.zip" | cut -f1)"
            part_num=$((part_num + 1))
        fi
    done
    
    # Remove the original large zip file
    rm -f "$zip_path"
    
    # Create a manifest file with information about the parts
    local manifest_path="${base_name}.manifest"
    {
        echo "# Multi-part zip manifest"
        echo "original_file: $(basename "$zip_path")"
        echo "total_parts: $((part_num - 1))"
        echo "split_size: 1900M"
        echo "created: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo ""
        echo "# To reconstruct the original file, use:"
        echo "# cat ${base_name}.part*.zip > $(basename "$zip_path")"
    } > "$manifest_path"
    
    echo "      📄 Created manifest: $(basename "$manifest_path")"
    
    return 0
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
    
    # Also remove any existing part files
    rm -f "${zip_path%.zip}".part*.zip
    rm -f "${zip_path%.zip}".manifest
    
    # Create zip file from the models directory
    cd "$MODELS_DIR"
    if [ -d "$model_path" ]; then
        zip -r "../$zip_path" "$model_path" -q
        cd ..
        
        local size=$(get_dir_size "$MODELS_DIR/$model_path")
        local zip_size=$(du -h "$zip_path" | cut -f1)
        
        echo "    ✅ Created: $model_type/$field_name/$zip_name (${size} -> ${zip_size})"
        
        # Check if the zip file needs to be split
        split_zip_file "$zip_path"
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

# Collect ML models in field-first layout: ml/<field>/<variation>
if [ -d "$MODELS_DIR/ml" ]; then
    for field_dir in "$MODELS_DIR/ml"/*; do
        if [ -d "$field_dir" ]; then
            field_name=$(basename "$field_dir")
            for variant_dir in "$field_dir"/*; do
                if [ -d "$variant_dir" ]; then
                    variant_name=$(basename "$variant_dir")
                    if matches_filters "$field_name" "$variant_name"; then
                        model_path="ml/$field_name/$variant_name"
                        MODEL_VARIANTS+=("$model_path|ml|$field_name|$variant_name")
                    fi
                fi
            done
        fi
    done
fi

# Collect LLM models in field-first layout: llm/<field>/<variation>
if [ -d "$MODELS_DIR/llm" ]; then
    for field_dir in "$MODELS_DIR/llm"/*; do
        if [ -d "$field_dir" ]; then
            field_name=$(basename "$field_dir")
            for variant_dir in "$field_dir"/*; do
                if [ -d "$variant_dir" ]; then
                    variant_name=$(basename "$variant_dir")
                    if matches_filters "$field_name" "$variant_name"; then
                        model_path="llm/$field_name/$variant_name"
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
        echo "  Fields: $(find "$MODELS_DIR" -mindepth 2 -maxdepth 2 -type d | sed 's:.*/::' | sort -u | tr '\n' ' ')" >&2
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
    echo "│   │   └── baseline.zip (or baseline.part*.zip for large files)"
    echo "│   ├── material/"
    echo "│   │   └── baseline.zip"
    echo "│   └── ..."
    echo "└── llm/"
    echo "    ├── category/"
    echo "    │   ├── baseline.zip (or baseline.part*.zip)"
    echo "    │   └── v1.zip"
    echo "    ├── material/"
    echo "    │   └── baseline.zip"
    echo "    └── ..."
    echo "\`\`\`"
    echo ""
    echo "## ML Models"
    if [ -d "$OUTPUT_DIR/ml" ]; then
        # Process each model, grouping multi-part files
        find "$OUTPUT_DIR/ml" -name "*.zip" -o -name "*.manifest" | sed 's/\.part[0-9]*\.zip$//' | sed 's/\.manifest$//' | sort -u | while read base_path; do
            if [ -f "${base_path}.manifest" ]; then
                # Multi-part file
                total_size=0
                parts_count=0
                for part in "${base_path}".part*.zip; do
                    if [ -f "$part" ]; then
                        part_size=$(stat -c%s "$part" 2>/dev/null || stat -f%z "$part" 2>/dev/null)
                        total_size=$((total_size + part_size))
                        parts_count=$((parts_count + 1))
                    fi
                done
                relative_path=$(echo "${base_path}.zip" | sed "s|$OUTPUT_DIR/||")
                total_size_human=$(echo "$total_size" | awk '{ split("B KB MB GB TB", units); for(i=1; $1>=1024 && i<5; i++) $1/=1024; printf "%.1f%s", $1, units[i] }')
                echo "- $relative_path ($total_size_human in $parts_count parts)"
            elif [ -f "${base_path}.zip" ]; then
                # Single file
                size=$(du -h "${base_path}.zip" | cut -f1)
                relative_path=$(echo "${base_path}.zip" | sed "s|$OUTPUT_DIR/||")
                echo "- $relative_path ($size)"
            fi
        done
    fi
    echo ""
    echo "## LLM Models"
    if [ -d "$OUTPUT_DIR/llm" ]; then
        # Process each model, grouping multi-part files
        find "$OUTPUT_DIR/llm" -name "*.zip" -o -name "*.manifest" | sed 's/\.part[0-9]*\.zip$//' | sed 's/\.manifest$//' | sort -u | while read base_path; do
            if [ -f "${base_path}.manifest" ]; then
                # Multi-part file
                total_size=0
                parts_count=0
                for part in "${base_path}".part*.zip; do
                    if [ -f "$part" ]; then
                        part_size=$(stat -c%s "$part" 2>/dev/null || stat -f%z "$part" 2>/dev/null)
                        total_size=$((total_size + part_size))
                        parts_count=$((parts_count + 1))
                    fi
                done
                relative_path=$(echo "${base_path}.zip" | sed "s|$OUTPUT_DIR/||")
                total_size_human=$(echo "$total_size" | awk '{ split("B KB MB GB TB", units); for(i=1; $1>=1024 && i<5; i++) $1/=1024; printf "%.1f%s", $1, units[i] }')
                echo "- $relative_path ($total_size_human in $parts_count parts)"
            elif [ -f "${base_path}.zip" ]; then
                # Single file
                size=$(du -h "${base_path}.zip" | cut -f1)
                relative_path=$(echo "${base_path}.zip" | sed "s|$OUTPUT_DIR/||")
                echo "- $relative_path ($size)"
            fi
        done
    fi
} > "$OUTPUT_DIR/README.md"

# Show final summary
echo ""
echo "🎯 Individual model zipping completed!"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Total files created: $(find "$OUTPUT_DIR" \( -name "*.zip" -o -name "*.manifest" \) | wc -l)"
echo "📄 Summary: $OUTPUT_DIR/README.md"

# Show directory structure
echo ""
echo "📁 Directory structure created:"
echo "  $OUTPUT_DIR/"
echo "  ├── ml/"
echo "  │   └── [field]/[model].zip (or .part*.zip + .manifest for large files)"
echo "  └── llm/"
echo "      └── [field]/[model].zip (or .part*.zip + .manifest for large files)"

# Show largest files and multi-part files
echo ""
echo "📏 File summary:"
# Count multi-part files
multi_part_count=$(find "$OUTPUT_DIR" -name "*.manifest" | wc -l)
if [ "$multi_part_count" -gt 0 ]; then
    echo "  🔀 Multi-part files: $multi_part_count"
    find "$OUTPUT_DIR" -name "*.manifest" | while read manifest; do
        base_name=$(basename "${manifest%.manifest}")
        dir_name=$(dirname "$manifest" | sed "s|$OUTPUT_DIR/||")
        parts_count=$(grep "total_parts:" "$manifest" | cut -d' ' -f2)
        echo "    - $dir_name/$base_name.zip (split into $parts_count parts)"
    done
fi

# Show single large files
echo ""
echo "  📦 Largest single files:"
find "$OUTPUT_DIR" -name "*.zip" | grep -v "\.part[0-9]*\.zip$" | xargs -I {} sh -c 'du -h "$1" | echo "$(cat) $1"' _ {} | sort -h | tail -5 | while read size file_path actual_file; do
    relative_path=$(echo "$actual_file" | sed "s|$OUTPUT_DIR/||")
    echo "    - $relative_path ($size)"
done
