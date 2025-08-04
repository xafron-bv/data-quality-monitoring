# Documentation Fixes Summary

This document summarizes all the discrepancies found between documentation and code, and the fixes applied.

## Environment Variables

### Issue
The documentation referenced environment variables that don't exist in the code:
- `DATA_QUALITY_DATA_PATH`
- `DATA_QUALITY_MODEL_PATH`
- `DATA_QUALITY_OUTPUT_PATH`
- `DQ_DATA_PATH`
- `DQ_OUTPUT_PATH`
- `DQ_LOG_LEVEL`
- `DQ_DEVICE`

### Fix
Removed these environment variable sections from:
- `/workspace/docs/getting-started/installation.md`
- `/workspace/docs/reference/cli.md`

## Command Line Interface

### Issue
Documentation showed direct Python script execution instead of using the main.py entrypoint.

### Fixes Applied

1. **Changed `python single_sample_multi_field_demo.py` to `python main.py single-demo`** in:
   - `/workspace/README.md`
   - `/workspace/docs/getting-started/quick-start.md`
   - `/workspace/docs/getting-started/basic-usage.md`
   - `/workspace/docs/getting-started/installation.md`
   - `/workspace/docs/detection-methods/overview.md`
   - `/workspace/docs/configuration/brand-config.md`

2. **Changed `python analyze_column.py` to `python main.py analyze-column`** in:
   - `/workspace/README.md`
   - `/workspace/docs/configuration/brand-config.md`

3. **Changed `python multi_sample_evaluation.py` to `python main.py multi-eval`** in:
   - `/workspace/README.md`
   - `/workspace/docs/development/new-fields.md`

4. **Changed `python ml_curve_generator.py` to `python main.py ml-curves`** in:
   - `/workspace/README.md`

5. **Changed ML training commands** in:
   - `/workspace/README.md`: `python anomaly_detectors/ml_based/model_training.py` → `python main.py ml-train`
   - `/workspace/anomaly_detectors/README.md`
   - `/workspace/anomaly_detectors/ml_based/README.md`

6. **Changed LLM training commands** in:
   - `/workspace/anomaly_detectors/llm_based/README.md`

## Command Arguments

### Issues Fixed

1. **analyze-column command**:
   - Removed non-existent arguments: `--list-columns`, `--column`, `--show-stats`, `--show-patterns`, `--sample-values`
   - Fixed to use positional arguments: `CSV_FILE` and optional `FIELD_NAME`

2. **ml-curves command**:
   - Changed from `--field` and `--data-file` to positional `DATA_FILE` argument
   - Fixed other arguments to match actual implementation
   - Removed non-existent `--metric` argument

3. **Removed references to non-existent scripts**:
   - `verify_field_detection.py` (removed from docs/development/new-fields.md)

## Other Fixes

1. **Fixed incorrect argument names**:
   - Changed `--pattern-threshold` to `--anomaly-threshold` in docs/detection-methods/overview.md

2. **Updated deprecated command options**:
   - Removed references to dynamic brand configuration options
   - Updated examples to reflect that brand configuration is now static

3. **Removed test mode flags that don't exist**:
   - Removed `--test-mode`, `--dry-run`, `--sample-size` (when used incorrectly)

## Summary

All documentation has been updated to accurately reflect the actual command-line interface and available options in the codebase. The main changes were:
1. Standardizing all commands to use the `main.py` entrypoint
2. Removing references to non-existent environment variables
3. Fixing command arguments to match the actual implementation
4. Removing references to deprecated or non-existent features