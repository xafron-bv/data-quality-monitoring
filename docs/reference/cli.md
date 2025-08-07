# CLI Reference

This document provides a concise reference for all command-line interfaces in the Data Quality Detection System.

## Main Entry Point

The system provides a unified entry point through `main.py`:

```bash
python main.py <command>
```

### Available Commands

- single-demo
- multi-eval
- ml-train
- llm-train
- analyze-column
- ml-curves

## Command Reference

### single-demo
Run single sample demonstration with comprehensive detection.

```bash
python main.py single-demo --help
```

#### Required
- `--data-file PATH`: Path to input CSV file (if brand config lacks a default)

#### Optional

- Output:
  - `--output-dir PATH` (default: `demo_results`)

- Detection Methods (disabled unless enabled):
  - `--enable-validation`
  - `--enable-pattern`
  - `--enable-ml`
  - `--enable-llm`

- Thresholds:
  - `--validation-threshold FLOAT` (default: 0.0)
  - `--anomaly-threshold FLOAT` (default: 0.7)
  - `--ml-threshold FLOAT` (default: 0.7)
  - `--llm-threshold FLOAT` (default: 0.6)

- Error Injection:
  - `--injection-intensity FLOAT` (default: 0.2)
  - `--max-issues-per-row INT` (default: 2)

- LLM Options:
  - `--llm-few-shot-examples`
  - `--llm-temporal-column STR`
  - `--llm-context-columns STR` (comma-separated)

- Combination Strategy:
  - `--use-weighted-combination`
  - `--weights-file PATH` (default: `detection_weights.json`)
  - `--generate-weights`
  - `--weights-output-file PATH`
  - `--baseline-weight FLOAT` (default: 0.1)

- Field Selection:
  - `--core-fields-only`

- Brand Selection:
  - If multiple brand configs exist, pass `--brand <name>`; otherwise the single brand is auto-selected

#### Outputs
- `<sample>_viewer_report.json`
- `<sample>_unified_report.json`
- PNG visualizations in the output directory

### multi-eval
Run evaluation across multiple samples for performance analysis.

```bash
python main.py multi-eval data/source.csv --field material --num-samples 50 --output-dir evaluation_results
```

#### Arguments
- Positional:
  - `source_data`: Path to the source CSV data file
- Required:
  - `--field FIELD`: Target field to validate (e.g., `material`)
- Optional:
  - `--validator STR` (defaults to field)
  - `--anomaly-detector STR` (defaults to validator)
  - `--ml-detector`
  - `--llm-detector`
  - `--run {validation,anomaly,ml,llm,both,all}` (default: both)
  - `--num-samples INT` (default: 32)
  - `--max-errors INT` (default: 3)
  - `--output-dir PATH` (default: `evaluation_results`)
  - `--ignore-errors ...`
  - `--ignore-fp`
  - Thresholds: `--validation-threshold`, `--anomaly-threshold`, `--ml-threshold`, `--llm-threshold`
  - Performance: `--batch-size`, `--max-workers`, `--high-confidence-threshold`

### ml-train
Train ML-based anomaly detection models or run anomaly checks.

```bash
python main.py ml-train data.csv --use-hp-search --fields "material color_name"
```

Key options:
- Positional: `csv_file`
- `--use-hp-search`, `--hp-trials INT`
- `--fields FIELD [FIELD ...]`
- `--check-anomalies FIELD`, `--threshold FLOAT`, `--output PATH`

### analyze-column
Analyze a specific column in a CSV file.

```bash
python main.py analyze-column data/products.csv material
```

Key options:
- Positional: `CSV_FILE`, `FIELD_NAME` (default: `color_name`)

### ml-curves
Generate precision-recall and ROC curves for ML-based and LLM-based anomaly detection.

```bash
python main.py ml-curves data/products.csv --detection-type ml --fields "material color_name"
```

Key options:
- Positional: `DATA_FILE`
- `--detection-type {ml,llm}` (default: ml)
- `--fields FIELD [FIELD ...]`
- `--output-dir PATH` (default: `detection_curves`)
- `--thresholds FLOAT [FLOAT ...]`

## Notes
- Methods are disabled by default unless `--enable-*` flags are set
- Single brand auto-selection occurs only when exactly one brand config is present
- Outputs of single-demo are JSON reports plus PNGs saved directly in `--output-dir`