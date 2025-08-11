# Data Directory Scripts

This directory contains scripts for managing data files, models, and GitHub releases.

## 📁 Directory Structure

```
data/
├── models/                          # Extracted models (created by decrypt/extract)
│   ├── ml/                          # ML models: data/models/ml/{field}/{variation}/
│   └── llm/                         # LLM models: data/models/llm/{field}/{variation}/
├── model_zips/                      # Individual model zip files (created by zip_models_individually.sh)
│   ├── ml/                          # ML model zips
│   │   ├── category/
│   │   │   └── baseline.zip
│   │   ├── material/
│   │   │   └── baseline.zip
│   │   └── ...
│   └── llm/                         # LLM model zips
│       ├── category/
│       │   ├── baseline.zip
│       │   └── v1.zip
│       ├── material/
│       │   └── baseline.zip
│       └── ...
├── encrypted_csv_files.zip          # Encrypted CSV files
├── encrypted_models.zip             # Encrypted models archive (optional)
└── *.csv                           # Decrypted CSV files
```

## 🛠️ Scripts

### Core Scripts

#### `encrypt.sh` - Encrypt CSV Files and Models Archive
```bash
./encrypt.sh <password>
```
- Encrypts CSV files into `encrypted_csv_files.zip`
- Encrypts `models/` into `encrypted_models.zip` if present

#### `decrypt.sh` - Decrypt Assets
```bash
./decrypt.sh <password>                    # Decrypt CSV and models zips if present
./decrypt.sh <password> encrypted_models.zip  # Decrypt specific zip
```
- Decrypts CSV and models archives if present

#### `extract_models.sh` - Extract Individual Models
```bash
./extract_models.sh                        # List available models
./extract_models.sh ml/category/baseline   # Extract specific model
./extract_models.sh -f category            # Extract all category models
./extract_models.sh -v baseline            # Extract all baseline variations
./extract_models.sh -f category -v baseline  # Extract category baseline models
```
- Lists all available models in `model_zips/` directory
- Extracts specific models to `models/` directory
- Supports filtering by field and/or variation
- Shows model sizes and extraction paths

### GitHub Integration

#### `release.sh` - Create GitHub Release
```bash
./release.sh [tag] [target_branch]
```
- Creates/updates GitHub release with encrypted CSV/models archives

#### `download.sh` - Download from GitHub Release
```bash
./download.sh <owner/repo> <tag> [asset1 asset2 ...]
```
- Downloads specified assets from GitHub releases into `data/`

### Model Management

#### `zip_models_individually.sh` - Create Individual Model Zips
```bash
./zip_models_individually.sh [options]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
```
- Creates individual zip files for each model variant
- Uses parallel processing (8 cores) for speed
- Creates organized directory structure: `model_zips/[type]/[field]/[variation].zip`
- All files under 2GB for GitHub compatibility
- Supports filtering by field and/or variation

## 🎯 Key Features

### ✅ Individual Model Zips
- Organized structure: `model_zips/ml/category/baseline.zip`, `model_zips/llm/material/baseline.zip`
- Parallel processing for faster creation

### ✅ Field and Variation Filtering
- Filter by field: `-f category` to process only category models
- Filter by variation: `-v baseline` to process only baseline variations
- Combined filtering: `-f category -v baseline` for category baseline models

### ✅ GitHub Integration
- Release encrypted CSV and models archives
- Download specific assets by name

## 🔧 Troubleshooting

### Common Issues
1. **Script not found**: Ensure you're in the `data/` directory
2. **Permission denied**: Run `chmod +x *.sh` to make scripts executable
3. **Large file upload fails**: Use individual model zips (under 2GB)
4. **Model not found**: Check `./extract_models.sh` for available models
5. **Filter returns no results**: Check available fields/variations with `./extract_models.sh -h`

### File Locations
- **Individual zips**: `model_zips/` directory (organized by type/field/variation)
- **Extracted models**: `models/` directory (ml/llm/{field}/{variation})
- **CSV files**: Current directory (after decryption)
- **Encrypted data**: `encrypted_csv_files.zip`, `encrypted_models.zip`
