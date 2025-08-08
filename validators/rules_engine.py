import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface


class JsonRulesValidator(ValidatorInterface):
    """
    Generic JSON rules-based validator.

    Loads rules from validators/{field_name}/rules.json and evaluates them in order.
    On first violation, returns a ValidationError; otherwise returns None.
    """

    class GenericErrorCode(str, Enum):
        INVALID_VALUE = "INVALID_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        INVALID_FORMAT = "INVALID_FORMAT"
        MISSING_VALUE = "MISSING_VALUE"

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.rules: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        rules_file = Path(__file__).parent / self.field_name / "rules.json"
        if not rules_file.exists():
            # Try to scaffold a default rules.json
            rules_file.parent.mkdir(parents=True, exist_ok=True)
            default = {
                "field_name": self.field_name,
                "description": f"Validation rules for {self.field_name}",
                "rule_flow": [
                    {
                        "name": "not_empty",
                        "type": "not_empty",
                        "error_type": str(self.GenericErrorCode.MISSING_VALUE),
                        "probability": 1.0
                    },
                    {
                        "name": "type_string",
                        "type": "type_is_string",
                        "allow_numeric_cast": False,
                        "error_type": str(self.GenericErrorCode.INVALID_TYPE),
                        "probability": 1.0
                    }
                ]
            }
            with open(rules_file, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
        with open(rules_file, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def _fail(self, error_type: Any, probability: float, details: Dict[str, Any], rule: Optional[Dict[str, Any]] = None) -> ValidationError:
        # Attach message template from the rule when available
        if rule and 'message' in rule:
            details = {**details, 'message': rule['message']}
        return ValidationError(error_type=error_type, probability=probability, details=details)

    def _ensure_string(self, value: Any, allow_numeric_cast: bool, rule: Optional[Dict[str, Any]] = None) -> Tuple[Optional[ValidationError], Any]:
        if isinstance(value, str):
            return None, value
        if allow_numeric_cast and isinstance(value, (int, float)):
            return None, str(value)
        return (
            self._fail(
                error_type=self.GenericErrorCode.INVALID_TYPE,
                probability=1.0,
                details={"expected": "string" if not allow_numeric_cast else "string or numeric", "received": str(type(value))},
                rule=rule,
            ),
            value,
        )

    def _tokenize(self, s: str, token_regex: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", s.strip())
        return re.compile(token_regex).findall(normalized)

    # Handlers for generic rule types
    def _handle_not_empty(self, value: Any, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()) or value == "":
            return self._fail(rule.get("error_type", self.GenericErrorCode.MISSING_VALUE), rule.get("probability", 1.0), rule.get("details", {}), rule=rule)
        return None

    def _handle_type_is_string(self, value: Any, rule: Dict[str, Any]) -> Tuple[Optional[ValidationError], Any]:
        allow_cast = bool(rule.get("allow_numeric_cast", False))
        return self._ensure_string(value, allow_cast, rule=rule)

    def _handle_no_leading_trailing_whitespace(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if value != value.strip():
            details = rule.get("details") or {"original": value, "stripped": value.strip()}
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.95), details, rule=rule)
        return None

    def _handle_no_line_breaks(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if "\n" in value or "\r" in value:
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_regex_forbidden(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        pattern = rule.get("pattern", "")
        flags = re.IGNORECASE if rule.get("ignore_case", False) else 0
        if pattern and re.search(pattern, value, flags):
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_regex_required(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        pattern = rule.get("pattern", "")
        if pattern and not re.search(pattern, value):
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.9), rule.get("details", {"value": value}), rule=rule)
        return None

    def _handle_must_match_one_of(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        patterns: List[str] = rule.get("patterns", [])
        if not any(re.match(p, value) for p in patterns):
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.8), rule.get("details", {"invalid_value": value}), rule=rule)
        return None

    # Tokenization and token-based material checks
    def _handle_tokenize(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        token_regex = rule.get("token_regex", r"(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])")
        tokens = self._tokenize(value, token_regex)
        self.context["tokens"] = tokens
        return None

    def _handle_tokens_not_empty(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        if not tokens:
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 1.0), rule.get("details", {}), rule=rule)
        return None

    def _handle_invalid_characters_tokens(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        allowed_regex = rule.get("allowed_chars_regex", r"[^a-zA-Z0-9\s%\-\/\(\):\.]")
        invalid_tokens = sorted({t for t in tokens if re.search(allowed_regex, t)})
        if invalid_tokens:
            details = rule.get("details") or {"display_message": f"Invalid characters detected: {invalid_tokens}", "chars": invalid_tokens}
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.8), details, rule=rule)
        return None

    def _handle_material_ambiguous_prefix(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1].isalpha():
            prefix = f"{tokens[0]} {tokens[1]}"
            details = rule.get("details") or {"display_message": f"Ambiguous prefix detected: '{prefix}'", "prefix": prefix}
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 0.6), details, rule=rule)
        return None

    def _handle_material_prepended_text(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1] != ":":
            details = rule.get("details") or {"display_message": f"Prepended text detected: '{tokens[0]}'", "text": tokens[0]}
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 0.7), details, rule=rule)
        return None

    def _handle_material_invalid_hyphen_delimiter(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        for i in range(len(tokens) - 1):
            if tokens[i].endswith('%') and tokens[i + 1] == '-':
                return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.8), rule.get("details", {}), rule=rule)
        return None

    def _handle_material_malformed_token(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        for i in range(len(tokens) - 2):
            is_num = re.fullmatch(r"\d+(?:\.\d+)?", tokens[i])
            is_word = tokens[i + 1].isalpha()
            is_percent = tokens[i + 2] == '%'
            if is_num and is_word and is_percent:
                token = f"{tokens[i]}{tokens[i + 1]}{tokens[i + 2]}"
                details = rule.get("details") or {"display_message": f"Malformed token detected: '{token}'", "token": token}
                return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.9), details, rule=rule)
        return None

    def _handle_material_missing_percentage_sign(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        for token in tokens:
            if re.fullmatch(r"\d+(?:\.\d+)?", token):
                details = rule.get("details") or {"display_message": f"Number '{token}' missing percentage sign", "number": token}
                return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.8), details, rule=rule)
        return None

    def _handle_token_duplicate_adjacent(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        for i in range(len(tokens) - 1):
            if tokens[i].isalpha() and tokens[i] == tokens[i + 1]:
                details = rule.get("details") or {"display_message": f"Duplicate material name: '{tokens[i]}'", "material": tokens[i]}
                return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 0.9), details, rule=rule)
        return None

    def _handle_material_missing_composition(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        numbers = [float(t.replace('%', '')) for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?%", t)]
        if not numbers:
            return self._fail(rule.get("error_type", self.GenericErrorCode.MISSING_VALUE), rule.get("probability", 1.0), rule.get("details", {}), rule=rule)
        self.context["percent_numbers"] = numbers
        return None

    def _handle_material_extraneous_text_appended(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        last_percent_indices = [i for i, token in enumerate(tokens) if re.fullmatch(r"\d+(?:\.\d+)?%", token)]
        if last_percent_indices:
            last_idx = last_percent_indices[-1]
            if len(tokens) > last_idx + 2:
                for i in range(last_idx + 2, len(tokens)):
                    if tokens[i].isalpha():
                        appended_text = " ".join(tokens[i:])
                        details = rule.get("details") or {"display_message": f"Extraneous text appended: '{appended_text}'", "text": appended_text}
                        return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 0.7), details, rule=rule)
        return None

    def _handle_material_sum_to_100(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        tokens: List[str] = self.context.get("tokens", [])
        numbers = [float(t.replace('%', '')) for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?%", t)]
        total_sum = sum(numbers)
        has_colon_delimiter = any(':' in t for t in tokens)
        allow_multi_parts = bool(rule.get("allow_colon_multi_parts", True))
        if abs(total_sum - 100.0) > 1e-6:
            is_invalid_multi_part = has_colon_delimiter and (total_sum <= 100 or abs(total_sum % 100.0) > 1e-6)
            if not (allow_multi_parts and has_colon_delimiter) or is_invalid_multi_part:
                details = rule.get("details") or {"display_message": f"Composition percentages sum to {total_sum}%, not 100%", "sum": total_sum}
                return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_VALUE), rule.get("probability", 0.9), details, rule=rule)
        return None

    # Season specific handler (parameterized)
    def _handle_season_validator(self, value: Any, rule: Dict[str, Any]) -> Optional[ValidationError]:
        # Missing check
        if pd.isna(value) or value == "":
            return self._fail(rule.get("missing_error_type", "MISSING_VALUE"), rule.get("missing_probability", 1.0), {}, rule=rule)
        # Type
        if not isinstance(value, str):
            return self._fail(rule.get("type_error_type", "INVALID_TYPE"), rule.get("type_probability", 1.0), {"expected": "string", "received": str(type(value))}, rule=rule)
        parts = value.strip().split(" ", 1)
        if len(parts) != 2:
            return self._fail(rule.get("format_error_type", "INVALID_FORMAT"), rule.get("format_probability", 0.95), {"value": value, "expected_format": "YYYY Season"}, rule=rule)
        year_str, season_name = parts
        if not year_str.isdigit() or len(year_str) != 4:
            return self._fail(rule.get("year_error_type", "INVALID_YEAR"), rule.get("year_probability", 0.9), {"year": year_str}, rule=rule)
        year = int(year_str)
        current_year = int(rule.get("current_year", 2025))
        max_offset = int(rule.get("max_future_offset", 5))
        if year < int(rule.get("min_year", 1900)) or year > current_year + max_offset:
            prob = rule.get("year_probability_out_of_range", 0.85 if year <= current_year + max_offset else 0.95)
            return self._fail(rule.get("year_error_type", "INVALID_YEAR"), prob, {"year": year, "min_year": int(rule.get("min_year", 1900)), "max_year": current_year + max_offset}, rule=rule)
        valid_seasons = rule.get("valid_seasons", ["Spring", "Summer", "Fall", "Winter", "Resort", "Holiday", "Pre-Fall", "Pre-Spring"])
        if season_name not in valid_seasons:
            return self._fail(rule.get("season_name_error_type", "INVALID_SEASON_NAME"), rule.get("season_name_probability", 0.9), {"season": season_name, "valid_seasons": valid_seasons}, rule=rule)
        return None

    # Category specific helper
    def _handle_html_tags_forbidden(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if re.search(r"<[^>]+>", value):
            return self._fail(rule.get("error_type", "HTML_TAGS"), rule.get("probability", 0.98), {"category": value}, rule=rule)
        return None

    def _handle_special_characters_forbidden(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        # default excludes alphanumeric, spaces, hyphens, underscores
        if re.search(rule.get("forbidden_regex", r"[^\w\s\-]"), value):
            return self._fail(rule.get("error_type", "SPECIAL_CHARACTERS"), rule.get("probability", 0.9), {"value": value}, rule=rule)
        return None

    def _handle_camelcase_merged_words(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if re.search(r"[a-z][A-Z]", value):
            suggested = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
            details = rule.get("details") or {"original": value, "suggested": suggested}
            return self._fail(rule.get("error_type", "MERGED_WORDS"), rule.get("probability", 0.85), details, rule=rule)
        return None

    def _handle_dashes_instead_of_slashes(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if " - " in value:
            corrected = value.replace(" - ", "/")
            details = rule.get("details") or {"original": value, "suggested": corrected}
            return self._fail(rule.get("error_type", "INVALID_FORMAT"), rule.get("probability", 0.8), details, rule=rule)
        return None

    # Care instructions handlers
    def _handle_starts_with(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        prefix = rule.get("prefix", "")
        if prefix and not value.startswith(prefix):
            return self._fail(rule.get("error_type", "CONTAINS_PREPENDED_TEXT"), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_contains_both(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        substr1 = rule.get("substr1", "")
        substr2 = rule.get("substr2", "")
        if substr1 in value and substr2 in value:
            return self._fail(rule.get("error_type", "CONTAINS_APPENDED_TEXT"), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_contains_multiple_spaces(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if "  " in value:
            return self._fail(rule.get("error_type", "CONTAINS_MULTIPLE_SPACES"), rule.get("probability", 0.85), rule.get("details", {}), rule=rule)
        return None

    def _handle_contains_html(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if re.search(r"<.*?>", value):
            return self._fail(rule.get("error_type", "CONTAINS_HTML"), rule.get("probability", 0.98), rule.get("details", {}), rule=rule)
        return None

    def _handle_contains_emoji(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF™®🧼]")
        if emoji_pattern.search(value):
            return self._fail(rule.get("error_type", "CONTAINS_EMOJI"), rule.get("probability", 0.95), rule.get("details", {}), rule=rule)
        return None

    def _handle_contains_disallowed_symbol(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        symbol = rule.get("symbol", "")
        if symbol and symbol in value:
            details = rule.get("details") or {"symbol": symbol}
            return self._fail(rule.get("error_type", "CONTAINS_DISALLOWED_SYMBOLS"), rule.get("probability", 0.9), details, rule=rule)
        return None

    def _handle_invalid_delimiter(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if ';' in value or (' - ' in value and 'NIET' in value):
            return self._fail(rule.get("error_type", "INVALID_DELIMITER"), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_missing_instruction(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        if '. .' in value or '..' in value:
            return self._fail(rule.get("error_type", "MISSING_INSTRUCTION"), rule.get("probability", 0.9), rule.get("details", {}), rule=rule)
        return None

    def _handle_regex_required_fullmatch(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        pattern = rule.get("pattern", "")
        if pattern and not re.fullmatch(pattern, value):
            return self._fail(rule.get("error_type", self.GenericErrorCode.INVALID_FORMAT), rule.get("probability", 0.9), rule.get("details", {"value": value}), rule=rule)
        return None

    def _handle_temperature_required(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        pattern = rule.get("pattern", r"WASSEN OP MAX (\d{2}°C)")
        if not re.search(pattern, value):
            details = rule.get("details") or {"value": value}
            return self._fail(rule.get("error_type", "INVALID_TEMPERATURE_FORMAT"), rule.get("probability", 0.9), details, rule=rule)
        return None

    def _handle_uppercase_instructions(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        parts = value.split('. ')
        for part in parts:
            if part and "WASSEN OP MAX" not in part and part != part.upper():
                details = rule.get("details") or {"instruction": part}
                return self._fail(rule.get("error_type", "INCORRECT_CAPITALIZATION"), rule.get("probability", 0.85), details, rule=rule)
        return None

    def _handle_allowed_instructions(self, value: str, rule: Dict[str, Any]) -> Optional[ValidationError]:
        allowed: List[str] = rule.get("allowed", [])
        cleaned = value.rstrip('.')
        parts = [p.strip() for p in cleaned.split('.') if p.strip()]
        for part in parts:
            if part not in allowed:
                if not re.fullmatch(r"WASSEN OP MAX \d{2}°C", part):
                    return self._fail(rule.get("error_type", "UNKNOWN_INSTRUCTION"), rule.get("probability", 0.8), {"instruction": part}, rule=rule)
        return None

    def _dispatch(self, rule_type: str):
        return {
            # generic
            "not_empty": self._handle_not_empty,
            "type_is_string": self._handle_type_is_string,
            "no_leading_trailing_whitespace": self._handle_no_leading_trailing_whitespace,
            "no_line_breaks": self._handle_no_line_breaks,
            "regex_forbidden": self._handle_regex_forbidden,
            "regex_required": self._handle_regex_required,
            "must_match_one_of": self._handle_must_match_one_of,
            # tokens/material
            "tokenize": self._handle_tokenize,
            "tokens_not_empty": self._handle_tokens_not_empty,
            "invalid_characters_tokens": self._handle_invalid_characters_tokens,
            "material_ambiguous_prefix": self._handle_material_ambiguous_prefix,
            "material_prepended_text": self._handle_material_prepended_text,
            "material_invalid_hyphen_delimiter": self._handle_material_invalid_hyphen_delimiter,
            "material_malformed_token": self._handle_material_malformed_token,
            "material_missing_percentage_sign": self._handle_material_missing_percentage_sign,
            "token_duplicate_adjacent": self._handle_token_duplicate_adjacent,
            "material_missing_composition": self._handle_material_missing_composition,
            "material_extraneous_text_appended": self._handle_material_extraneous_text_appended,
            "material_sum_to_100": self._handle_material_sum_to_100,
            # season
            "season_validator": self._handle_season_validator,
            # category/color_name helpers
            "html_tags_forbidden": self._handle_html_tags_forbidden,
            "special_characters_forbidden": self._handle_special_characters_forbidden,
            "camelcase_merged_words": self._handle_camelcase_merged_words,
            "dashes_instead_of_slashes": self._handle_dashes_instead_of_slashes,
            # care instructions
            "starts_with": self._handle_starts_with,
            "contains_both": self._handle_contains_both,
            "contains_multiple_spaces": self._handle_contains_multiple_spaces,
            "contains_html": self._handle_contains_html,
            "contains_emoji": self._handle_contains_emoji,
            "contains_disallowed_symbol": self._handle_contains_disallowed_symbol,
            "invalid_delimiter": self._handle_invalid_delimiter,
            "missing_instruction": self._handle_missing_instruction,
            "regex_required_fullmatch": self._handle_regex_required_fullmatch,
            "temperature_required": self._handle_temperature_required,
            "uppercase_instructions": self._handle_uppercase_instructions,
            "allowed_instructions": self._handle_allowed_instructions,
        }.get(rule_type)

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        rule_flow: List[Dict[str, Any]] = self.rules.get("rule_flow", [])
        current_value = value
        for rule in rule_flow:
            rtype = rule.get("type")
            handler = self._dispatch(rtype)
            if handler is None:
                # Unknown rule, skip
                continue
            if rtype == "type_is_string":
                error, current_value = handler(current_value, rule)  # type: ignore[arg-type]
                if error:
                    return error
                # Keep normalized string (if any change) for subsequent rules
                value_str = current_value if isinstance(current_value, str) else str(current_value)
            else:
                value_str = current_value if isinstance(current_value, str) else str(current_value)
                result = handler(value_str, rule)
                if isinstance(result, ValidationError):
                    return result
                # Some handlers update context; no change to value_str unless we add normalization in future
        return None