#!/usr/bin/env bash
# Download helper for GitHub release assets into the data directory
# Usage: ./download.sh [options] <owner/repo> <tag> [asset1 asset2 ...]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
# Examples:
#   ./download.sh xafron-bv/data-quality-monitoring data-20250809               # downloads default zips
#   ./download.sh xafron-bv/data-quality-monitoring data-20250809 models.zip    # one asset
#   ./download.sh xafron-bv/data-quality-monitoring data-20250809 ml/category/baseline.zip  # specific model
#   ./download.sh -f category xafron-bv/data-quality-monitoring data-20250809   # download all category models
#   ./download.sh -v baseline xafron-bv/data-quality-monitoring data-20250809   # download all baseline variations

set -euo pipefail

# Parse command line arguments
FIELD_FILTER=""
VARIATION_FILTER=""
REPO=""
TAG=""
ASSETS=()

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
            echo "Usage: $0 [options] <owner/repo> <tag> [asset1 asset2 ...]"
            echo ""
            echo "Options:"
            echo "  -f, --field FIELD          Filter by field name (e.g., category, material)"
            echo "  -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 xafron-bv/data-quality-monitoring data-20250809           # downloads default zips"
            echo "  $0 xafron-bv/data-quality-monitoring data-20250809 models.zip # one asset"
            echo "  $0 xafron-bv/data-quality-monitoring data-20250809 ml/category/baseline.zip # specific model"
            echo "  $0 -f category xafron-bv/data-quality-monitoring data-20250809 # download all category models"
            echo "  $0 -v baseline xafron-bv/data-quality-monitoring data-20250809 # download all baseline variations"
            echo "  $0 -f category -v baseline xafron-bv/data-quality-monitoring data-20250809 # download category baseline models"
            echo ""
            echo "Note: This script uses 'gh CLI' for authentication. Make sure you're logged in with 'gh auth login'"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information" >&2
            exit 1
            ;;
        *)
            if [ -z "$REPO" ]; then
                REPO="$1"
            elif [ -z "$TAG" ]; then
                TAG="$1"
            else
                ASSETS+=("$1")
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$REPO" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 [options] <owner/repo> <tag> [asset1 asset2 ...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 xafron-bv/data-quality-monitoring data-20250809           # downloads default zips" >&2
    echo "  $0 xafron-bv/data-quality-monitoring data-20250809 models.zip # one asset" >&2
    echo "  $0 xafron-bv/data-quality-monitoring data-20250809 ml/category/baseline.zip # specific model" >&2
    echo "  $0 -f category xafron-bv/data-quality-monitoring data-20250809 # download all category models" >&2
    echo "  $0 -v baseline xafron-bv/data-quality-monitoring data-20250809 # download all baseline variations" >&2
    echo "  $0 -f category -v baseline xafron-bv/data-quality-monitoring data-20250809 # download category baseline models" >&2
    echo "" >&2
    echo "Note: This script uses 'gh CLI' for authentication. Make sure you're logged in with 'gh auth login'" >&2
    exit 1
fi

# Ensure running from data dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if gh CLI is available and authenticated
if ! command -v gh &> /dev/null; then
    echo "❌ Error: 'gh CLI' is not installed or not in PATH" >&2
    echo "Please install GitHub CLI: https://cli.github.com/" >&2
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub CLI" >&2
    echo "Please run 'gh auth login' to authenticate" >&2
    exit 1
fi

# Function to check if model matches filters
matches_filters() {
    local asset="$1"
    
    # Extract field and variation from asset path
    local field_name=""
    local variant_name=""
    
    # Handle both old and new naming conventions
    # Parse model path: ml/category/baseline.zip, llm/category/baseline.zip
    # or new format: ml_category_baseline.zip, llm_category_baseline.zip
    # Also handle multi-part files: ml_category_baseline.part1.zip, ml_category_baseline.manifest
    if [[ "$asset" =~ ^(ml|llm)/([^/]+)/([^/]+)\.(zip|part[0-9]+\.zip|manifest)$ ]]; then
        # Old format: ml/category/baseline.zip
        field_name="${BASH_REMATCH[2]}"
        variant_name="${BASH_REMATCH[3]}"
    elif [[ "$asset" =~ ^(ml|llm)_([^_]+)_([^_]+)\.(zip|part[0-9]+\.zip|manifest)$ ]]; then
        # New format: ml_category_baseline.zip or ml_category_baseline.part1.zip
        field_name="${BASH_REMATCH[2]}"
        variant_name="${BASH_REMATCH[3]}"
    else
        # Not a model file, include if no filters or if it matches
        return 0
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

# Function to download and merge multi-part files
download_and_merge_parts() {
    local base_name="$1"
    local output_path="$2"
    local manifest_name="${base_name}.manifest"
    
    echo "  🔍 Checking for multi-part file: $base_name"
    
    # First, try to download the manifest
    local manifest_path="${output_path%.zip}.manifest"
    if gh release download "$TAG" --repo "$REPO" --pattern "$manifest_name" --output "$manifest_path" 2>/dev/null; then
        echo "  📄 Found manifest file, this is a multi-part download"
        
        # Read the number of parts from manifest
        local total_parts=$(grep "total_parts:" "$manifest_path" | cut -d' ' -f2)
        echo "  📦 Total parts to download: $total_parts"
        
        # Download all parts
        local all_parts_success=true
        for i in $(seq 1 "$total_parts"); do
            local part_name="${base_name}.part${i}.zip"
            local part_path="${output_path%.zip}.part${i}.zip"
            
            echo "  ⬇️  Downloading part $i/$total_parts: $part_name..."
            if gh release download "$TAG" --repo "$REPO" --pattern "$part_name" --output "$part_path" 2>/dev/null; then
                echo "    ✅ Downloaded: $(du -h "$part_path" | cut -f1)"
            else
                echo "    ❌ Failed to download part $i"
                all_parts_success=false
                break
            fi
        done
        
        if [ "$all_parts_success" = true ]; then
            # Merge all parts
            echo "  🔀 Merging ${total_parts} parts into $output_path..."
            cat "${output_path%.zip}".part*.zip > "$output_path"
            
            # Verify the merged file exists and has content
            if [ -f "$output_path" ] && [ -s "$output_path" ]; then
                echo "  ✅ Successfully merged: $(du -h "$output_path" | cut -f1)"
                
                # Clean up part files
                echo "  🧹 Cleaning up part files..."
                rm -f "${output_path%.zip}".part*.zip
                
                return 0
            else
                echo "  ❌ Failed to merge parts"
                rm -f "$output_path"
                return 1
            fi
        else
            # Clean up any downloaded parts if we failed
            rm -f "${output_path%.zip}".part*.zip
            rm -f "$manifest_path"
            return 1
        fi
    else
        # No manifest found, this is not a multi-part file
        return 1
    fi
}

# If no specific assets provided and filters are specified, try to discover available models
if [ ${#ASSETS[@]} -eq 0 ] && ([ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]); then
    echo "🔍 Discovering available models with filters..."
    [ -n "$FIELD_FILTER" ] && echo "  - Field: $FIELD_FILTER"
    [ -n "$VARIATION_FILTER" ] && echo "  - Variation: $VARIATION_FILTER"
    
    # Try to get list of assets from the release using gh CLI
    if gh release view "$TAG" --repo "$REPO" --json assets >/dev/null 2>&1; then
        # Extract asset names from the release using gh CLI
        ASSET_NAMES=$(gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name')
        
        # Group multi-part files together
        declare -A seen_base_names
        
        # Filter assets based on criteria
        for asset_name in $ASSET_NAMES; do
            # Skip part files and manifests when building the main list
            if [[ "$asset_name" =~ \.part[0-9]+\.zip$ ]] || [[ "$asset_name" =~ \.manifest$ ]]; then
                continue
            fi
            
            # Check if this is the base file of a multi-part set
            base_name="${asset_name%.zip}"
            
            # Check if manifest exists for this file (indicating it's multi-part)
            if echo "$ASSET_NAMES" | grep -q "^${base_name}.manifest$"; then
                # This is a multi-part file, add only if not seen
                if [ -z "${seen_base_names[$base_name]:-}" ]; then
                    if matches_filters "$asset_name"; then
                        ASSETS+=("$asset_name")
                        seen_base_names[$base_name]=1
                    fi
                fi
            else
                # Regular single file
                if matches_filters "$asset_name"; then
                    ASSETS+=("$asset_name")
                fi
            fi
        done
        
        if [ ${#ASSETS[@]} -eq 0 ]; then
            echo "❌ Error: No assets found matching the specified filters." >&2
            echo "Available assets:" >&2
            echo "$ASSET_NAMES" | grep -v "\.part[0-9]*\.zip$" | grep -v "\.manifest$" | head -10
            if [ "$(echo "$ASSET_NAMES" | grep -v "\.part[0-9]*\.zip$" | grep -v "\.manifest$" | wc -l)" -gt 10 ]; then
                echo "  ... and $(($(echo "$ASSET_NAMES" | grep -v "\.part[0-9]*\.zip$" | grep -v "\.manifest$" | wc -l) - 10)) more"
            fi
            exit 1
        fi
        
        echo "📊 Found ${#ASSETS[@]} matching models to download"
    else
        echo "❌ Error: Could not fetch release information for $REPO@$TAG" >&2
        echo "💡 Please check that the repository and tag exist, and you have access to them" >&2
        exit 1
    fi
fi

# Default assets if none provided
if [ ${#ASSETS[@]} -eq 0 ]; then
    echo "❌ Error: No assets specified and no default assets available" >&2
    echo "💡 Please specify assets to download:" >&2
    echo "   $0 $REPO $TAG encrypted_csv_files.zip" >&2
    echo "   $0 $REPO $TAG ml/category/baseline.zip llm/material/baseline.zip" >&2
    exit 1
fi

# Download assets
for ASSET in "${ASSETS[@]}"; do
    echo "⬇️  Downloading $ASSET from $REPO@$TAG ..."
    
    # Create model_zips directory structure if downloading individual models
    if [[ "$ASSET" == ml/* ]] || [[ "$ASSET" == llm/* ]]; then
        # Create the directory structure for model zips
        asset_dir=$(dirname "model_zips/$ASSET")
        mkdir -p "$asset_dir"
        OUTPUT_PATH="model_zips/$ASSET"
    else
        OUTPUT_PATH="$ASSET"
    fi
    
    # Skip if this is a part file or manifest (these are handled by the main file)
    if [[ "$ASSET" =~ \.part[0-9]+\.zip$ ]] || [[ "$ASSET" =~ \.manifest$ ]]; then
        continue
    fi
    
    # Try to download as a multi-part file first
    base_name="${ASSET%.zip}"
    if download_and_merge_parts "$base_name" "$OUTPUT_PATH"; then
        # Successfully downloaded and merged multi-part file
        continue
    fi
    
    # If not a multi-part file, download normally
    if gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --output "$OUTPUT_PATH"; then
        echo "✅ Saved: $OUTPUT_PATH ($(du -h "$OUTPUT_PATH" | cut -f1))"
    else
        echo "❌ Failed to download $ASSET"
        if [ -f "$OUTPUT_PATH" ]; then
            rm -f "$OUTPUT_PATH"
        fi
    fi
done

echo "📦 Files in data/:"
ls -lah | sed -n '1,100p'

# Show model_zips if it exists
if [ -d "model_zips" ] && [ "$(find model_zips -name "*.zip" | wc -l)" -gt 0 ]; then
    echo ""
    echo "📁 Individual model zips in model_zips/:"
    find model_zips -name "*.zip" | head -10
    if [ "$(find model_zips -name "*.zip" | wc -l)" -gt 10 ]; then
        echo "  ... and $(($(find model_zips -name "*.zip" | wc -l) - 10)) more"
    fi
fi