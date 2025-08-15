#!/bin/bash
# Test script for verifying split/merge functionality

set -e

echo "🧪 Testing split/merge functionality..."

# Create a test directory
TEST_DIR="test_split_merge_$(date +%s)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create a large test file (2.1GB)
echo "📝 Creating test file (2.1GB)..."
dd if=/dev/zero of=large_file.bin bs=1M count=2150 2>/dev/null

# Create a zip file
echo "📦 Creating zip file..."
zip -q large_test.zip large_file.bin

# Get original checksum
ORIGINAL_CHECKSUM=$(md5sum large_file.bin | cut -d' ' -f1)
echo "🔐 Original checksum: $ORIGINAL_CHECKSUM"

# Test splitting
echo "🔀 Testing split functionality..."
MAX_FILE_SIZE=$((2000 * 1024 * 1024))  # 2000 MB
zip_size=$(stat -c%s large_test.zip 2>/dev/null || stat -f%z large_test.zip 2>/dev/null)

if [ "$zip_size" -gt "$MAX_FILE_SIZE" ]; then
    echo "  Splitting into 1900MB parts..."
    split -b 1900M large_test.zip large_test.part
    
    # Rename parts
    part_num=1
    for part in large_test.part*; do
        if [ -f "$part" ]; then
            mv "$part" "large_test.part${part_num}.zip"
            echo "  Created part ${part_num}: $(du -h "large_test.part${part_num}.zip" | cut -f1)"
            part_num=$((part_num + 1))
        fi
    done
    
    # Create manifest
    {
        echo "# Multi-part zip manifest"
        echo "original_file: large_test.zip"
        echo "total_parts: $((part_num - 1))"
        echo "split_size: 1900M"
        echo "created: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    } > large_test.manifest
    
    # Remove original
    rm -f large_test.zip
fi

# Test merging
echo "🔄 Testing merge functionality..."
if [ -f "large_test.manifest" ]; then
    total_parts=$(grep "total_parts:" large_test.manifest | cut -d' ' -f2)
    echo "  Merging $total_parts parts..."
    
    cat large_test.part*.zip > large_test_merged.zip
    
    # Extract and verify
    echo "📂 Extracting merged file..."
    unzip -q large_test_merged.zip
    
    # Check checksum
    MERGED_CHECKSUM=$(md5sum large_file.bin | cut -d' ' -f1)
    echo "🔐 Merged checksum: $MERGED_CHECKSUM"
    
    if [ "$ORIGINAL_CHECKSUM" = "$MERGED_CHECKSUM" ]; then
        echo "✅ Success! Checksums match - split/merge works correctly"
    else
        echo "❌ Error! Checksums don't match"
        echo "  Original: $ORIGINAL_CHECKSUM"
        echo "  Merged:   $MERGED_CHECKSUM"
    fi
fi

# Cleanup
cd ..
echo "🧹 Cleaning up test directory..."
rm -rf "$TEST_DIR"

echo "✅ Test completed!"