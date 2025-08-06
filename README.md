# Data Quality Detection System

A comprehensive, multi-method data quality monitoring system for detecting errors and anomalies in structured data, with a focus on fashion/retail product catalogs.

## Overview

This system combines multiple detection approaches to identify data quality issues:

- **Validation (Rule-Based)**: High-confidence error detection using business rules
- **Pattern-Based Anomaly Detection**: Medium-confidence detection using pattern matching
- **ML-Based Detection**: Semantic similarity analysis using sentence transformers
- **LLM-Based Detection**: Advanced semantic understanding with language models

The system is designed to be field-agnostic and brand-independent, making it adaptable to various data domains.

## Key Features

- 🎯 **Multi-Method Detection**: Combines rule-based, pattern-based, and ML approaches
- 📊 **Comprehensive Evaluation**: Built-in metrics and confusion matrix analysis
- 🔧 **Configurable**: Field mappings and detection thresholds are customizable
- 💾 **Memory Efficient**: Sequential processing and model caching
- 📈 **Performance Optimization**: Weighted combination based on historical performance
- 🌐 **Brand Agnostic**: Supports multiple brands through configuration
- 📱 **Visual Interface**: HTML5 viewer for interactive result exploration

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster ML/LLM detection)
- 8GB+ RAM recommended

### Quick Installation

```bash
# Clone and setup
git clone <repository-url>
cd <project-directory>
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For detailed installation instructions, GPU setup, and troubleshooting, see [Installation Guide](docs/getting-started/installation.md).

5. Configure brand settings:
```bash
# Create or edit brand configuration files in brand_configs/
# Example: brand_configs/esqualo.json for Esqualo brand
# Each brand should have its own JSON file in this directory
```

## Quick Start

### Running the Demo

The easiest way to see the system in action:

```bash
# Basic demo with all detection methods
python main.py single-demo \
    --data-file your_data.csv \
    --enable-validation \
    --enable-pattern \
    --enable-ml \
    --enable-llm
```

### Quick Start

```bash
# Run your first detection
python main.py single-demo --data-file your_data.csv

# View results in the web interface
# Open single_sample_multi_field_demo/data_quality_viewer.html
```

For detailed guides, see the [Documentation](docs/).

## Documentation

For comprehensive guides on configuration, usage, and development:

📚 **[View Full Documentation](docs/)**

- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start Tutorial](docs/getting-started/quick-start.md)  
- [Basic Usage Guide](docs/getting-started/basic-usage.md)
- [Configuration Reference](docs/reference/configuration.md)
- [CLI Reference](docs/reference/cli.md)
- [Development Guide](docs/development/adding-fields.md)

## Architecture

The system follows a modular, layered architecture:

- **Entry Points**: User-facing scripts for different use cases
- **Orchestration**: Coordinates detection methods and manages workflow
- **Detection Methods**: Independent implementations of each detection approach
- **Core Services**: Shared utilities for configuration, mapping, and reporting
- **Data Layer**: Handles data I/O and storage

## Performance Considerations

- **Memory Usage**: The system processes fields sequentially to minimize memory footprint
- **GPU Acceleration**: ML and LLM detection can utilize GPU if available
- **Caching**: Models are cached to avoid redundant loading
- **Batch Processing**: Configurable batch sizes for optimal performance

## Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use `--core-fields-only` flag
   - Reduce batch size
   - Disable ML/LLM detection

2. **Slow Performance**:
   - Enable GPU acceleration
   - Use parallel processing
   - Optimize detection thresholds

3. **Missing Fields**:
   - Check field mappings in your brand configuration file (e.g., `brand_configs/esqualo.json`)
   - Verify column names in your data

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details on:
- Development setup
- Code quality standards
- Submission process

## License

[Specify your license here]

## Support

For issues, questions, or contributions, please [specify contact method].