# Changelog

All notable changes to the Data Quality Monitoring System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Weighted Combination Detection System**: New approach for combining multiple anomaly detection methods using performance-based weights
  - `generate_detection_weights.py`: Tool to generate field-specific weights from evaluation results
  - Field-specific weight calculation based on F1 scores, precision, and recall metrics
  - JSON-based weight configuration for maintainable weight management
  - Automatic fallback to equal weights when weights file is unavailable
  - CLI support for weighted combination via `--use-weighted-combination` flag
  - Support for custom weights file via `--weights-file` parameter

### Changed
- **ComprehensiveFieldDetector**: Enhanced to support both priority-based and weighted combination approaches
  - Added `use_weighted_combination` parameter to constructor
  - Added `weights_file` parameter for custom weights file path
  - Modified `classify_cells()` to route to appropriate classification method
  - Added `_weighted_classify_cells()` method for weighted combination logic
  - Added `_load_field_weights_from_file()` method for loading weights from JSON
  - Added `_get_default_weights()` method for fallback behavior

- **Demo Script (demo.py)**: Updated to support weighted combination
  - Added `--use-weighted-combination` CLI argument
  - Added `--weights-file` CLI argument with default value
  - Updated constructor to accept weighted combination parameters
  - Enhanced output to show combination method being used

### Documentation
- Added comprehensive [Weighted Combination Documentation](./docs/WEIGHTED_COMBINATION.md)
- Updated README.md with weighted combination overview and quick start guide
- Added detection combination methods section to main documentation
- Documented new CLI arguments and usage patterns

### Implementation Details
- **Weight Calculation Algorithm**: 
  - F1-score based weight assignment with configurable baseline for untrained methods
  - Normalization to ensure weights sum to 1.0 per field
  - Performance extraction from demo evaluation results
- **Detection Logic**: 
  - Validation results maintain highest priority regardless of combination method
  - Weighted confidence scores combined using field-specific weights
  - Configurable anomaly threshold (default: 0.3) for weighted classification
- **Error Handling**: 
  - Graceful fallback to equal weights when weights file is missing or corrupted
  - Validation of weights file structure and format
  - Informative error messages and warnings

### Benefits
- **Improved Accuracy**: Dynamically prioritizes the most effective detection method for each field
- **Maintainable**: Externalized weight configuration eliminates hardcoded performance data
- **Flexible**: Supports easy updates when new models are trained or performance changes
- **Backward Compatible**: Existing priority-based approach remains the default

### Migration Notes
- Existing installations continue to use priority-based approach by default
- No breaking changes to existing API or CLI interfaces
- Weighted combination is opt-in via CLI flag
- Weights file generation requires running evaluation with ground truth data

---

## [Previous Versions]

Previous changes were not formally tracked in this changelog. The weighted combination feature represents the first major documented enhancement to the detection system.