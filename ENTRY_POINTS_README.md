# Entry Points Organization

This document describes the reorganization of entry point scripts into dedicated folders with their associated viewers and documentation.

## Structure Overview

Each entry point script has been moved to its own directory containing:
- The main script
- Associated HTML viewer(s)
- HOW-TO.md documentation
- Output files (generated in subdirectories)

## Entry Point Directories

### 1. `demo_analysis/`
**Main Script:** `single_sample_multi_field_demo.py`
- **Purpose:** Comprehensive demo of data quality monitoring with error injection and detection
- **Viewers:** 
  - `demo_report_viewer.html` - Main demo report viewer
  - `confusion_matrix_viewer.html` - Confusion matrix analysis viewer
- **Output:** `demo_results/` directory
- **Documentation:** [HOW-TO.md](demo_analysis/HOW-TO.md)

### 2. `ml_evaluation/`
**Main Script:** `multi_sample_evaluation.py`
- **Purpose:** Systematic evaluation of detection methods with multiple samples
- **Viewer:** `unified_report_viewer.html`
- **Output:** `evaluation_results/` directory
- **Documentation:** [HOW-TO.md](ml_evaluation/HOW-TO.md)

### 3. `weights_generation/`
**Main Script:** `generate_detection_weights.py`
- **Purpose:** Generate optimized detection weights from performance results
- **Viewer:** `detection_weights_viewer.html`
- **Output:** `detection_weights.json`
- **Documentation:** [HOW-TO.md](weights_generation/HOW-TO.md)

### 4. `detection_comparison/`
**Main Script:** `detection_comparison.py`
- **Purpose:** Compare ML and LLM detection methods side-by-side
- **Viewer:** `ml_summary_viewer.html`
- **Output:** `detection_comparison_results/` directory
- **Documentation:** [HOW-TO.md](detection_comparison/HOW-TO.md)

### 5. `column_analysis/`
**Main Script:** `analyze_column.py`
- **Purpose:** Analyze value distribution in CSV columns
- **Viewer:** None (console output or text file)
- **Output:** Console or specified file
- **Documentation:** [HOW-TO.md](column_analysis/HOW-TO.md)

## Quick Start Guide

### Running a Demo Analysis:
```bash
cd demo_analysis
python single_sample_multi_field_demo.py ../data/sample_data.csv
# View results in demo_report_viewer.html
```

### Evaluating Detection Methods:
```bash
cd ml_evaluation
python multi_sample_evaluation.py ../data/clean_data.csv size
# View results in unified_report_viewer.html
```

### Generating Detection Weights:
```bash
cd weights_generation
python generate_detection_weights.py -i ../demo_analysis/demo_results/demo_analysis_unified_report.json
# View results in detection_weights_viewer.html
```

### Comparing Detection Methods:
```bash
cd detection_comparison
python detection_comparison.py ../data/products.csv
# View results in ml_summary_viewer.html
```

### Analyzing Column Data:
```bash
cd column_analysis
python analyze_column.py ../data/products.csv material
```

## Viewer Access

All viewers can be accessed through the main index page:
```bash
# Open in browser
report_viewers_index.html
```

Or navigate directly to specific viewers in their respective directories.

## Import Path Updates

All scripts have been updated to import from the parent directory:
```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This ensures proper imports of shared modules like:
- `field_mapper`
- `brand_config`
- `exceptions`
- `comprehensive_detector`
- etc.

## Output Directory Behavior

Each script now saves outputs within its own directory by default:
- `demo_analysis/demo_results/`
- `ml_evaluation/evaluation_results/`
- `weights_generation/detection_weights.json`
- `detection_comparison/detection_comparison_results/`

This keeps related files organized and prevents cluttering the root directory.

## Migration Notes

### For Existing Users:
1. Scripts are no longer in the root directory
2. Navigate to the appropriate subdirectory before running
3. Update any automation scripts with new paths
4. Existing output directories in root can be moved or deleted

### For New Users:
1. Start with the demo in `demo_analysis/`
2. Read the HOW-TO.md in each directory
3. Use relative paths (e.g., `../data/`) for input files

## Benefits of This Organization

1. **Clarity:** Each tool has its own space with related files
2. **Documentation:** HOW-TO guides are with their tools
3. **Isolation:** Output files don't mix between tools
4. **Discoverability:** Viewers are with their generators
5. **Maintenance:** Easier to update individual tools

## Future Additions

New entry point scripts should follow the same pattern:
1. Create a new directory
2. Move/create the script
3. Add appropriate viewer(s)
4. Write a HOW-TO.md
5. Update import paths
6. Set default output to script directory