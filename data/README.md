# Data Directory Scripts

This directory contains scripts for managing data files, models, and GitHub releases.

## 📁 Directory Structure

```
data/
├── models/                          # Extracted models (created by decrypt/extract)
│   ├── ml/trained/                  # ML models
│   └── llm/                         # LLM models
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
├── models.zip                       # Combined models (legacy, large file)
└── *.csv                           # Decrypted CSV files
```

## 🛠️ Scripts

### Core Scripts

#### `encrypt.sh` - Encrypt CSV Files Only
```bash
./encrypt.sh <password>
```
- Encrypts CSV files into `encrypted_csv_files.zip`
- Only handles CSV encryption (models handled separately)

#### `decrypt.sh` - Decrypt CSV Files Only
```bash
./decrypt.sh <password>                    # Decrypt CSV files
./decrypt.sh <password> encrypted_csv_files.zip  # Decrypt specific CSV zip
```
- Decrypts CSV files from `encrypted_csv_files.zip`
- Only handles CSV decryption (models handled separately)

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
./release.sh [options] [tag] [target_branch]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
```
- Creates GitHub release with all assets
- Automatically includes all individual model zips from `model_zips/`
- Supports filtering by field and/or variation
- Falls back to `models.zip` if individual zips not found
- Uploads CSV and model assets to GitHub releases

#### `download.sh` - Download from GitHub Release
```bash
./download.sh [options] <owner/repo> <tag> [asset1 asset2 ...]
# Options:
#   -f, --field FIELD          Filter by field name (e.g., category, material)
#   -v, --variation VARIATION  Filter by variation name (e.g., baseline, v1)
#   -h, --help                 Show this help message
```
- Downloads assets from GitHub releases
- Automatically places model zips in `model_zips/` directory structure
- Supports downloading specific models
- Supports filtering by field and/or variation

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
- **29 separate zip files** instead of one large 5.3GB file
- **All under 2GB** for GitHub compatibility
- **Organized structure**: `model_zips/ml/category/baseline.zip`, `model_zips/llm/material/baseline.zip`
- **Parallel processing** for faster creation

### ✅ Field and Variation Filtering
- **Filter by field**: `-f category` to process only category models
- **Filter by variation**: `-v baseline` to process only baseline variations
- **Combined filtering**: `-f category -v baseline` for category baseline models
- **Available in all scripts**: zip, extract, release, download

### ✅ Backward Compatibility
- Still supports legacy `models.zip` structure
- All existing scripts work with both old and new formats
- Automatic detection of available formats

### ✅ GitHub Integration
- **Automatic asset detection** in release script
- **Individual downloads** for specific models
- **Organized structure** for easy management

## 📊 Model Statistics

### ML Models (13 files)
- **Size range**: 80MB - 387MB
- **Total size**: ~2.1GB
- **Fields**: category, color_name, material, size, season, care_instructions, ean, article_number, description_short_1, product_name_en, long_description_nl, customs_tariff_number, colour_code

### LLM Models (16 files)
- **Size range**: 236MB - 236MB
- **Total size**: ~3.8GB
- **Fields**: category, color_name, material, size, season, care_instructions, ean, article_number, description_short_1, product_name_en, long_description_nl, customs_tariff_number, colour_code, brand

## 🚀 Usage Examples

### For Contributors
```bash
# 1. Encrypt data and create individual model zips
./encrypt.sh mypassword
./zip_models_individually.sh

# 2. Create GitHub release with all models
./release.sh data-20250809

# 3. Create GitHub release with only category models
./release.sh -f category data-20250809

# 4. Create GitHub release with only baseline variations
./release.sh -v baseline data-20250809

# 5. Download specific models
./download.sh owner/repo data-20250809 ml/category/baseline.zip
```

### For Users
```bash
# 1. Download and extract specific models
./download.sh owner/repo data-20250809 ml/category/baseline.zip
./extract_models.sh ml/category/baseline

# 2. Download and extract all category models
./download.sh -f category owner/repo data-20250809
./extract_models.sh -f category

# 3. Download and extract all baseline variations
./download.sh -v baseline owner/repo data-20250809
./extract_models.sh -v baseline

# 4. Or extract all models (if models.zip exists)
./decrypt.sh mypassword models.zip
```

## 🔧 Filtering Options

All scripts that handle models support the following filtering options:

### Field Filtering (`-f, --field`)
Filter models by field name:
```bash
# Only category models
./zip_models_individually.sh -f category

# Only material models
./extract_models.sh -f material

# Only color_name models
./release.sh -f color_name data-20250809
```

### Variation Filtering (`-v, --variation`)
Filter models by variation name:
```bash
# Only baseline variations
./zip_models_individually.sh -v baseline

# Only v1 variations
./extract_models.sh -v v1

# Only baseline variations
./download.sh -v baseline owner/repo data-20250809
```

### Combined Filtering
Combine both field and variation filters:
```bash
# Only category baseline models
./zip_models_individually.sh -f category -v baseline

# Extract category baseline models
./extract_models.sh -f category -v baseline

# Release category baseline models
./release.sh -f category -v baseline data-20250809
```

## 🔧 Troubleshooting

### Common Issues
1. **Script not found**: Ensure you're in the `data/` directory
2. **Permission denied**: Run `chmod +x *.sh` to make scripts executable
3. **Large file upload fails**: Use individual model zips (under 2GB)
4. **Model not found**: Check `./extract_models.sh` for available models
5. **Filter returns no results**: Check available fields/variations with `./extract_models.sh -h`

### File Locations
- **Individual zips**: `model_zips/` directory (organized by type/field/variation)
- **Extracted models**: `models/` directory
- **CSV files**: Current directory (after decryption)
- **Encrypted data**: `encrypted_csv_files.zip`
