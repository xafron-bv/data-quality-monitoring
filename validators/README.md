# Validators

This directory contains field-specific validators and a generic JSON rules engine.

- Each field lives under `validators/{field_name}`
- Validation rules are configured in `validators/{field_name}/rules.json`
- Human-readable error descriptions remain in `validators/{field_name}/error_messages.json`

## JSON Rules Engine

Validators now delegate to a generic engine in `validators/rules_engine.py` which loads and applies rules from `rules.json` in order. The first rule that fails produces a `ValidationError`.

Basic structure of `rules.json`:

```json
{
  "field_name": "material",
  "description": "Human description",
  "rule_flow": [
    {"name": "not_empty", "type": "not_empty", "error_type": "MISSING_VALUE", "probability": 1.0},
    {"name": "type_string", "type": "type_is_string", "allow_numeric_cast": false, "error_type": "INVALID_TYPE", "probability": 1.0}
  ]
}
```

Supported rule types include:
- not_empty
- type_is_string (option: `allow_numeric_cast`)
- no_leading_trailing_whitespace
- no_line_breaks
- regex_forbidden (keys: `pattern`, `ignore_case`)
- regex_required (key: `pattern`)
- must_match_one_of (key: `patterns`)
- tokenize (key: `token_regex`)
- tokens_not_empty
- invalid_characters_tokens (key: `allowed_chars_regex`)
- material_* family: ambiguous_prefix, prepended_text, invalid_hyphen_delimiter, malformed_token, missing_percentage_sign, missing_composition, extraneous_text_appended, sum_to_100, token_duplicate_adjacent
- season_validator (parameterized end-to-end season checks)
- html_tags_forbidden, special_characters_forbidden, camelcase_merged_words, dashes_instead_of_slashes
- care-instructions: starts_with, contains_both, contains_multiple_spaces, contains_html, contains_emoji, contains_disallowed_symbol, invalid_delimiter, missing_instruction, regex_required_fullmatch, temperature_required, uppercase_instructions, allowed_instructions

See examples in existing fields' `rules.json` files.

## Creating a New Field

1. Create a new directory `validators/{field_name}`
2. Create `rules.json` describing your rule flow
3. Create `validate.py` with:
```python
from typing import Any, Optional
from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator

class Validator(ValidatorInterface):
    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="{field_name}")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        return self._engine._validate_entry(value)
```
4. Optionally add `error_messages.json` mapping error types to messages.

## Backwards Compatibility

Existing validators have been migrated to JSON rules and now simply delegate to the engine. You can extend rule types in `validators/rules_engine.py` if new logic is needed.