# Validators Module

High-confidence, rule-based validation for data quality monitoring.

## Overview

Validation is now fully configuration-driven. Define rules per field in JSON files; no Python validator classes are required.

- Rule-based validator engine: `validators/rule_based/rule_based_validator.py`
- Rules location: `validators/rule_based/rules/{field}.json`
- Reporter: `validators/report.py` (generic messages; optional per-field overrides not required)

## Rule Schema

See `validators/rule_based/rules/README.md` for the full schema and examples.

Minimal example:

```json
{
  "field_name": "material",
  "known_values": [],
  "format_patterns": [
    {"name": "chars", "pattern": "^[A-Za-z0-9\\s%/().,-]+$", "message": "Invalid characters"}
  ],
  "validation_rules": [
    {"name": "not_empty", "type": "not_empty", "message": "Value cannot be empty"}
  ]
}
```

## Creating a New Field

1. Create `validators/rule_based/rules/{field}.json`. If missing, a template will be auto-generated on first run.
2. Populate `known_values`, `format_patterns`, and `validation_rules`.
3. Run detection with `--enable-validation`.

## Available Fields (JSON Rules)

- `material` (rules provided)
- `size` (rules provided)
- `season` (rules provided)
- `category` (rules provided)
- `color_name` (rules provided)
- `care_instructions` (rules provided)

## Best Practices

- Keep rules deterministic and high-confidence
- Use meaningful `message` text for user-friendly reports
- Prefer `regex_must_match`/`regex_must_not_match` for complex patterns
- Use `known_values` when the domain is closed
