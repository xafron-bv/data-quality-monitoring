# Basic Usage

This guide shows how to run validation and anomaly detection.

## Validation (Rule-Based)

Create a rule file for your field if not present:

- Location: `validators/rule_based/rules/{field}.json`
- A template is auto-generated on first run. No Python validators are needed.

Run the single-sample demo with validation enabled:

```bash
python main.py single-demo --brand esqualo --enable-validation
```

## Pattern-Based Anomaly Detection

Rules live under `anomaly_detectors/pattern_based/rules/{field}.json`.

Enable in the demo:

```bash
python main.py single-demo --brand esqualo --enable-pattern
```

## ML-Based Detection

```bash
python main.py single-demo --brand esqualo --enable-ml
```

