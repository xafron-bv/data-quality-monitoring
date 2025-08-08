#!/bin/bash
# Script to decrypt CSV files only
# Usage: ./decrypt.sh <password> [zip_file]

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <password> [zip_file]"
    echo "This script will decrypt CSV files with password protection."
    echo "For model extraction, use: ./extract_models.sh"
    echo ""
    echo "Examples:"
    echo "  ./decrypt.sh <password>                    # decrypt CSV files"
    echo "  ./decrypt.sh <password> encrypted_csv_files.zip  # decrypt specific CSV zip"
    exit 1
fi

PASSWORD="$1"
ZIP_FILE="$2"

# Check if we're in the data directory
if [ ! -f "decrypt.sh" ]; then
    echo "Error: This script must be run from the data directory"
    exit 1
fi

# If a specific archive is provided, process only that
if [ -n "$ZIP_FILE" ]; then
  if [ ! -f "$ZIP_FILE" ]; then
      echo "Error: Zip file '$ZIP_FILE' not found"
      exit 1
  fi
  
  echo "Processing $ZIP_FILE..."
  if [[ "$ZIP_FILE" == *"encrypted"* ]]; then
    # Encrypted file - use password
    unzip -o -P "$PASSWORD" "$ZIP_FILE"
    echo "✅ Decrypted: $ZIP_FILE"
  else
    echo "❌ File '$ZIP_FILE' doesn't appear to be encrypted"
    exit 1
  fi
  exit 0
fi

# Otherwise, attempt standard CSV archive
CSV_ZIP_FILE="encrypted_csv_files.zip"

if [ -f "$CSV_ZIP_FILE" ]; then
  echo "Decrypting CSV files from $CSV_ZIP_FILE..."
  unzip -o -P "$PASSWORD" "$CSV_ZIP_FILE"
  echo "📁 CSV files present:"
  ls -la *.csv 2>/dev/null || echo "No CSV files found in zip"
else
  echo "No $CSV_ZIP_FILE found. Skipping CSVs."
fi

echo ""
echo "💡 To extract models, run: ./extract_models.sh"
