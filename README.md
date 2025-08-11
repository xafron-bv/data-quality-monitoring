# Data Quality Monitoring

A comprehensive, multi-method data quality monitoring system for detecting errors and anomalies in structured data.

## Overview

This system combines multiple detection approaches to identify data quality issues with varying confidence levels:

- **Validation (Rule-Based)**: High-confidence error detection using business rules
- **Pattern-Based Detection**: Medium-confidence detection using pattern matching
- **ML-Based Detection**: Semantic similarity analysis using sentence transformers
- **LLM-Based Detection**: Advanced semantic understanding with language models

## Key Features

- 🎯 **Multi-Method Detection**: Combines rule-based, pattern-based, and ML approaches
- 📊 **Comprehensive Evaluation**: Built-in metrics and performance analysis
- 🔧 **Highly Configurable**: Customizable field mappings and detection thresholds
- 📈 **Performance Optimization**: Weighted combination based on historical data
- 📱 **Visual Interface**: Interactive HTML viewer for result exploration

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd data-quality-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Run detection (example with esqualo brand)
python single_sample_multi_field_demo/single_sample_multi_field_demo.py --brand esqualo --data-file your_data.csv

# View results
# Open single_sample_multi_field_demo/data_quality_viewer.html
```

## Documentation

📚 **[View Full Documentation](docs/)**

The comprehensive documentation includes:
- Installation and setup guides
- Usage tutorials and examples
- Configuration reference
- Architecture documentation
- Development guides

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

## License

[Specify your license here]

## Support

For issues, questions, or contributions, please [specify contact method].

## Large data handling (GitHub Releases)

Helpers in `data/`:
- Encrypt: `./data/encrypt.sh <password>` → creates `encrypted_csv_files.zip` and `encrypted_models.zip`
- Release: `./data/release.sh [tag] [target_branch]` (requires `gh` CLI) → creates/updates a release and uploads the zips
- Download: `./data/download.sh <owner/repo> <tag> [asset1 asset2 ...]` → fetches release assets into `data/`
- Decrypt: `./data/decrypt.sh <password> [zip_file]` → extracts the assets

Models live under `data/models/` (and are gitignored):
- ML: `data/models/ml/{field}/{variation}/`
- LLM: `data/models/llm/{field}/{variation}/`