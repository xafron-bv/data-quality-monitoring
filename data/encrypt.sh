#!/bin/bash
# Script to encrypt CSV files only
# Usage: ./encrypt.sh <password>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <password>"
    echo "This script will encrypt CSV files with password protection."
    echo "For model zipping, use: ./zip_models_individually.sh"
    exit 1
fi

PASSWORD="$1"
CSV_ZIP_FILE="encrypted_csv_files.zip"

# Check if we're in the data directory
if [ ! -f "encrypt.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

# Encrypt CSV files if present
if ls *.csv >/dev/null 2>&1; then
  echo "Encrypting CSV files with password protection..."
  [ -f "$CSV_ZIP_FILE" ] && rm -f "$CSV_ZIP_FILE" && echo "Removed existing $CSV_ZIP_FILE"
  zip -e -P "$PASSWORD" "$CSV_ZIP_FILE" *.csv
  echo "✅ Encrypted CSV -> $CSV_ZIP_FILE ($(du -h "$CSV_ZIP_FILE" | cut -f1))"
else
  echo "No CSV files found to encrypt. Skipping CSV archive."
fi

echo ""
echo "💡 To create model zips, run: ./zip_models_individually.sh"
