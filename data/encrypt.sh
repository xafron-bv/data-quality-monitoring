#!/bin/bash
# Script to encrypt CSV files using zip with password protection
# Usage: ./encrypt.sh <password>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <password>"
    echo "This script will encrypt all CSV files in the current directory using zip with password protection."
    exit 1
fi

PASSWORD="$1"
ZIP_FILE="encrypted_csv_files.zip"

# Check if we're in the data directory
if [ ! -f "encrypt.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

# Check if there are any CSV files
if ! ls *.csv >/dev/null 2>&1; then
    echo "No CSV files found in the current directory"
    exit 1
fi

echo "Encrypting CSV files with password protection..."

# Remove existing zip file if it exists
if [ -f "$ZIP_FILE" ]; then
    rm "$ZIP_FILE"
    echo "Removed existing $ZIP_FILE"
fi

# Create encrypted zip file
# Using -P for password and -e for encryption
zip -e -P "$PASSWORD" "$ZIP_FILE" *.csv

if [ $? -eq 0 ]; then
    echo "✅ Successfully encrypted CSV files into $ZIP_FILE"
    echo "📊 File size: $(du -h "$ZIP_FILE" | cut -f1)"
else
    echo "❌ Failed to encrypt files"
    exit 1
fi
