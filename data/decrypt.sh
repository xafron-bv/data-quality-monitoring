#!/bin/bash
# Script to decrypt CSV files using unzip with password protection
# Usage: ./decrypt.sh <password> [zip_file]

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <password> [zip_file]"
    echo "This script will decrypt CSV files from a password-protected zip file."
    echo "If zip_file is not specified, defaults to encrypted_csv_files.zip"
    exit 1
fi

PASSWORD="$1"
ZIP_FILE="${2:-encrypted_csv_files.zip}"

# Check if we're in the data directory
if [ ! -f "decrypt.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

# Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Zip file '$ZIP_FILE' not found"
    exit 1
fi

echo "Decrypting CSV files from $ZIP_FILE..."

# Extract files using unzip with password
# Using -P for password
unzip -P "$PASSWORD" "$ZIP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Successfully decrypted CSV files from $ZIP_FILE"
    echo "📁 Extracted files:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found in zip"
else
    echo "❌ Failed to decrypt files"
    echo "This might be due to an incorrect password or corrupted zip file"
    exit 1
fi
