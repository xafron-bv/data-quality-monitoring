# Rule-Based Validation Rules

JSON schema for validators in `validators/rule_based/rules/{field}.json`.

Example:

```json
{
  "field_name": "material",
  "description": "Validation rules for material",
  "known_values": ["cotton", "polyester"],
  "format_patterns": [
    {"name": "chars", "pattern": "^[A-Za-z0-9\\s%/().,-]+$", "message": "Invalid characters"}
  ],
  "validation_rules": [
    {"name": "not_empty", "type": "not_empty", "message": "Value cannot be empty"},
    {"name": "maxlen", "type": "max_length", "max_length": 120, "message": "Too long"},
    {"name": "must_contain_pct", "type": "must_contain", "substring": "%", "message": "Missing %"}
  ]
}
```

Supported rule types:
- not_empty
- max_length (max_length)
- min_length (min_length)
- numeric_range (min, max)
- must_contain (substring)
- must_not_contain (substring)
- regex_must_match (pattern)
- regex_must_not_match (pattern)