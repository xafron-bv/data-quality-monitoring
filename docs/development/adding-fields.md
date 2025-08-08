# Adding New Fields Guide

This guide walks you through adding support for new fields. Validation is JSON-rule based; no Python validator class is needed.

## Overview

Adding a new field involves:
1. Creating validation rules (JSON)
2. Defining pattern-based detection rules (optional)
3. Training ML models (optional)
4. Configuring field mappings
5. Testing the implementation

## Step 1: Analyze the Field

Understand your field's characteristics:

```bash
python main.py analyze-column data/sample.csv new_field_name
```

## Step 2: Create Validation Rules (JSON)

Create `validators/rule_based/rules/new_field.json`:

```json
{
  "field_name": "new_field",
  "description": "Validation rules for new field",
  "known_values": [],
  "format_patterns": [
    {"name": "standard_format", "pattern": "^[A-Z]{2}\\d{4}$", "message": "Does not match standard format"}
  ],
  "validation_rules": [
    {"name": "not_empty", "type": "not_empty", "message": "Value cannot be empty"},
    {"name": "length_min", "type": "min_length", "min_length": 6, "message": "Too short"},
    {"name": "length_max", "type": "max_length", "max_length": 50, "message": "Too long"}
  ]
}
```

Supported rule types are documented in `validators/rule_based/rules/README.md`.

## Step 3: Pattern-Based Rules (Optional)

Create `anomaly_detectors/pattern_based/rules/new_field.json` if you also want pattern-based anomaly detection.

## Step 4: Train ML Model (Optional)

Train an ML model for semantic anomalies if needed (see `ml-train`).

## Step 5: Configure Field Mapping

Update brand configuration to include the new field so pipelines can locate the column.

## Step 6: Test

```bash
python main.py single-demo --brand your_brand --enable-validation --output-dir out
```

If you also defined pattern/ML detection:

```bash
python main.py single-demo --brand your_brand --enable-validation --enable-pattern --enable-ml --output-dir out
```

## Best Practices

- Start simple; iterate rules based on false positives/negatives
- Use `message` to provide clear, actionable feedback
- Favor `regex_must_match`/`regex_must_not_match` for complex formats
- Use `known_values` for closed vocabularies
