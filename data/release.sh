#!/usr/bin/env bash
# Release helper for publishing encrypted CSV and zipped model archives to GitHub Releases
# Requires: GitHub CLI (gh) authenticated (gh auth login)
# Usage:
#   ./release.sh [options] [tag] [target_branch]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
# Examples:
#   ./release.sh                 # tag=data-YYYYMMDD, target=repo default or main
#   ./release.sh data-20250809   # explicit tag, auto target
#   ./release.sh data-20250809 main
#   ./release.sh -f category data-20250809  # only category models
#   ./release.sh -v baseline data-20250809  # only baseline variations

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI (gh) not found. Install from https://cli.github.com/" >&2
  exit 1
fi

# Parse command line arguments
FIELD_FILTER=""
VARIATION_FILTER=""
TAG=""
TARGET_BRANCH=""

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
            echo "Usage: $0 [options] [tag] [target_branch]"
            echo "Options:"
            echo "  -f, --field FIELD          Filter by field name (e.g., category, material)"
            echo "  -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # tag=data-YYYYMMDD, target=repo default or main"
            echo "  $0 data-20250809                     # explicit tag, auto target"
            echo "  $0 data-20250809 main               # explicit tag and target"
            echo "  $0 -f category data-20250809        # only category models"
            echo "  $0 -v baseline data-20250809        # only baseline variations"
            echo "  $0 -f category -v baseline data-20250809  # only category baseline"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$TAG" ]; then
                TAG="$1"
            elif [ -z "$TARGET_BRANCH" ]; then
                TARGET_BRANCH="$1"
            else
                echo "Too many arguments provided"
                exit 1
            fi
            shift
            ;;
    esac
done

# Default tag = today's date
DEFAULT_TAG="data-$(date +%Y%m%d)"
TAG="${TAG:-$DEFAULT_TAG}"

# Determine default branch from origin; do not assume main if unknown
DETECTED_BRANCH="$(git remote show origin 2>/dev/null | sed -n '/HEAD branch/s/.*: //p')"
if [ -z "$DETECTED_BRANCH" ] && [ -z "${2:-}" ]; then
  echo "❌ Could not detect default branch from 'origin'. Please specify target branch: ./release.sh [tag] <target_branch>" >&2
  exit 1
fi
DEFAULT_BRANCH="${DETECTED_BRANCH:-}" 
TARGET_BRANCH="${2:-$DEFAULT_BRANCH}"
if [ -z "$TARGET_BRANCH" ]; then
  echo "❌ Target branch is required when default cannot be detected." >&2
  exit 1
fi

TITLE="Data assets $TAG"
BODY="Encrypted CSV files and individual model zips (models not encrypted for speed)"

# Add filter info to body if filters are applied
if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
    FILTER_INFO="\n\nFilters applied:"
    [ -n "$FIELD_FILTER" ] && FILTER_INFO="$FILTER_INFO\n- Field: $FIELD_FILTER"
    [ -n "$VARIATION_FILTER" ] && FILTER_INFO="$FILTER_INFO\n- Variation: $VARIATION_FILTER"
    BODY="$BODY$FILTER_INFO"
fi

# Ensure running from data dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ASSETS=()
[ -f encrypted_csv_files.zip ] && ASSETS+=( encrypted_csv_files.zip )

# Function to check if model matches filters
matches_filters() {
    local zip_file="$1"
    local model_path=""
    
    # Handle both old and new naming conventions
    local relative_path=$(echo "$zip_file" | sed "s|model_zips/||")
    
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
    
    # Extract field and variation from model path
    local field_name=""
    local variant_name=""
    
    # Parse model path: ml/category/baseline or llm/category/baseline
    if [[ "$model_path" =~ ^(ml|llm)/([^/]+)/([^/]+)$ ]]; then
        field_name="${BASH_REMATCH[2]}"
        variant_name="${BASH_REMATCH[3]}"
    else
        return 0  # Not a model file, include it
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

# Check for individual model zips first (preferred)
if [ -d "model_zips" ] && [ "$(find model_zips -name "*.zip" | wc -l)" -gt 0 ]; then
  echo "📁 Found individual model zips in model_zips/ directory"
  echo "These are ready for GitHub releases (each under 2GB limit)"
  
  # Apply filters if specified
  if [ -n "$FIELD_FILTER" ] || [ -n "$VARIATION_FILTER" ]; then
    echo "🔍 Applying filters:"
    [ -n "$FIELD_FILTER" ] && echo "  - Field: $FIELD_FILTER"
    [ -n "$VARIATION_FILTER" ] && echo "  - Variation: $VARIATION_FILTER"
  fi
  
  # Add filtered model zips
  while IFS= read -r -d '' zip_file; do
    if [ -z "$FIELD_FILTER" ] && [ -z "$VARIATION_FILTER" ]; then
      # No filters, include all
      ASSETS+=( "$zip_file" )
    elif matches_filters "$zip_file"; then
      # Matches filters, include
      ASSETS+=( "$zip_file" )
    fi
  done < <(find model_zips -name "*.zip" -print0)
  
  echo "📊 Will upload ${#ASSETS[@]} assets (CSV + $((${#ASSETS[@]} - 1)) model zips)"
else
  echo "❌ Error: Individual model zips not found in model_zips/ directory" >&2
  echo "💡 Please run ./zip_models_individually.sh first to create individual model zips" >&2
  exit 1
fi

if [ ${#ASSETS[@]} -eq 0 ]; then
  echo "⚠️  No zip assets found in data/. Run ./encrypt.sh <password> first." >&2
  exit 1
fi

echo "ℹ️  Using tag: $TAG"
echo "ℹ️  Target branch: $TARGET_BRANCH"

# Create release if it doesn't exist; otherwise continue
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "ℹ️  Release $TAG exists. Uploading assets..."
else
  echo "🚀 Creating release $TAG on $TARGET_BRANCH ..."
  gh release create "$TAG" -t "$TITLE" -n "$BODY" --target "$TARGET_BRANCH"
fi

echo "📤 Uploading assets: ${ASSETS[*]}"
# Upload each asset individually to handle errors gracefully
for asset in "${ASSETS[@]}"; do
  echo "  📤 Uploading $asset..."
  if gh release upload "$TAG" "$asset" --clobber 2>/dev/null; then
    echo "    ✅ Successfully uploaded $asset"
  else
    echo "    ⚠️  Failed to upload $asset (may already exist or be too large)"
    # Try without clobber for new assets
    if gh release upload "$TAG" "$asset" 2>/dev/null; then
      echo "    ✅ Successfully uploaded $asset (new asset)"
    else
      echo "    ❌ Failed to upload $asset"
    fi
  fi
done

echo "✅ Release published:"
gh release view "$TAG" --web || true