# Setting up the environment

## Install dev requirements

```bash
pip install -r requirements-dev.txt
```

## Decrypt the data and models

### Option 1: Extract all models (if models.zip exists)
```bash
./decrypt.sh <password>           # decrypts encrypted_csv_files.zip
# For models (if models.zip exists):
./extract_models.sh               # list available models
./extract_models.sh -f category   # extract all category models
./extract_models.sh -v baseline   # extract all baseline variations
```

### Option 2: Extract individual models (recommended)
```bash
# List available models
./extract_models.sh

# Extract specific models
./extract_models.sh ml/category/baseline
./extract_models.sh llm/material/baseline

# Extract models by field
./extract_models.sh -f category
./extract_models.sh -f material

# Extract models by variation
./extract_models.sh -v baseline
./extract_models.sh -v v1

# Extract models by field and variation
./extract_models.sh -f category -v baseline
```

## Encrypt the data and zip models (for contributors)

```bash
# Encrypt CSV files only
./encrypt.sh <password>           # creates encrypted_csv_files.zip

# Create model zips (all or filtered)
./zip_models_individually.sh      # creates all individual model zips
./zip_models_individually.sh -f category      # only category models
./zip_models_individually.sh -v baseline      # only baseline variations
./zip_models_individually.sh -f category -v baseline  # only category baseline
```

## Create GitHub releases (for contributors)

```bash
# Release all assets
./release.sh data-20250809

# Release specific models by field
./release.sh -f category data-20250809

# Release specific models by variation
./release.sh -v baseline data-20250809

# Release specific models by field and variation
./release.sh -f category -v baseline data-20250809
```

## Download from GitHub releases (for users)

```bash
# Download CSV only
./download.sh owner/repo data-20250809

# Download specific models
./download.sh owner/repo data-20250809 ml/category/baseline.zip

# Download models by field
./download.sh -f category owner/repo data-20250809

# Download models by variation
./download.sh -v baseline owner/repo data-20250809

# Download models by field and variation
./download.sh -f category -v baseline owner/repo data-20250809
```

## Notes

- **CSV files**: Handled by `encrypt.sh` and `decrypt.sh` only
- **Models**: Handled by `zip_models_individually.sh` and `extract_models.sh` only
- Models are stored under `data/models/`:
  - ML: `data/models/ml/{field}/{variation}/`
  - LLM: `data/models/llm/{field}/{variation}/`
- Individual model zips are stored in `data/model_zips/` with organized structure:
  - ML models: `data/model_zips/ml/{field}/{variation}.zip`
  - LLM models: `data/model_zips/llm/{field}/{variation}.zip`
- The scripts use native `zip`/`unzip` with password protection for CSV files only
- Models are zipped without encryption for faster processing
- Individual model zips are under 2GB for GitHub compatibility
- All scripts support filtering by field (`-f`) and/or variation (`-v`)
- Ensure scripts are executable: `chmod +x encrypt.sh decrypt.sh extract_models.sh zip_models_individually.sh release.sh download.sh`
- Original CSV files and `models/` are excluded from Git via `.gitignore` in `data/`
