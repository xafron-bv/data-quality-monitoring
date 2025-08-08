#!/usr/bin/env python3
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface


class RuleBasedValidator(ValidatorInterface):
    """
    Generic rule-based validator that loads field-specific rules from JSON files.

    Rules files are located in: validators/rule_based/rules/{field_name}.json
    If a rules file is missing, a default template is created to guide authors.
    """

    class ErrorCode(str, Enum):
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        INVALID_FORMAT = "INVALID_FORMAT"
        VALUE_NOT_ALLOWED = "VALUE_NOT_ALLOWED"
        RULE_VIOLATION = "RULE_VIOLATION"

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.rules: Dict[str, Any] = {}
        self.known_values: List[str] = []
        self.format_patterns: List[Dict[str, Any]] = []
        self.validation_rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _rules_dir(self) -> Path:
        return Path(__file__).parent / "rules"

    def _rules_file(self) -> Path:
        return self._rules_dir() / f"{self.field_name}.json"

    def _load_rules(self) -> None:
        rules_file = self._rules_file()
        if not rules_file.exists():
            self._create_default_rules_file(rules_file)
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
        except Exception as e:
            # Fall back to empty rules to avoid crashing validation
            self.rules = {}
        # Normalize and extract components
        raw_values = self.rules.get("known_values", []) if isinstance(self.rules, dict) else []
        self.known_values = [v.lower().strip() for v in raw_values if isinstance(v, str) and v and not v.strip().startswith("#")]
        self.format_patterns = self.rules.get("format_patterns", []) if isinstance(self.rules, dict) else []
        self.validation_rules = self.rules.get("validation_rules", []) if isinstance(self.rules, dict) else []

    def _create_default_rules_file(self, rules_file: Path) -> None:
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        default_rules = {
            "field_name": self.field_name,
            "description": f"Validation rules for {self.field_name} field",
            "known_values": [
                "# Add known valid values here (case-insensitive)",
                "# Example: valid_option_1, valid_option_2"
            ],
            "format_patterns": [
                {
                    "name": "basic_format",
                    "description": "Basic format validation",
                    "pattern": "^[\\w\\s\-_%/().,:]+$",
                    "message": "Contains invalid characters"
                }
            ],
            "validation_rules": [
                {
                    "name": "not_empty",
                    "description": "Value should not be empty",
                    "type": "not_empty",
                    "message": "Value cannot be empty"
                },
                {
                    "name": "max_length",
                    "description": "Value should not be too long",
                    "type": "max_length",
                    "max_length": 200,
                    "message": "Value is too long"
                }
            ]
        }
        try:
            with open(rules_file, "w", encoding="utf-8") as f:
                json.dump(default_rules, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _check_format_patterns(self, value: str) -> Optional[ValidationError]:
        for pattern_rule in self.format_patterns:
            pattern = pattern_rule.get("pattern", "")
            if not pattern:
                continue
            if not re.match(pattern, value):
                return ValidationError(
                    error_type=self.ErrorCode.INVALID_FORMAT,
                    probability=1.0,
                    details={
                        "field": self.field_name,
                        "value": value,
                        "rule": pattern_rule.get("name", "unknown"),
                        "message": pattern_rule.get("message", "Invalid format"),
                        "pattern": pattern,
                    },
                )
        return None

    def _check_validation_rules(self, value: str) -> Optional[ValidationError]:
        for rule in self.validation_rules:
            rule_type = rule.get("type", "")
            if rule_type == "not_empty":
                if not value.strip():
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "not_empty"),
                            "message": rule.get("message", "Value cannot be empty"),
                        },
                    )
            elif rule_type == "max_length":
                max_len = int(rule.get("max_length", 200))
                if len(value) > max_len:
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "max_length"),
                            "message": rule.get("message", "Value is too long"),
                            "actual_length": len(value),
                            "max_length": max_len,
                        },
                    )
            elif rule_type == "min_length":
                min_len = int(rule.get("min_length", 1))
                if len(value) < min_len:
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "min_length"),
                            "message": rule.get("message", "Value is too short"),
                            "actual_length": len(value),
                            "min_length": min_len,
                        },
                    )
            elif rule_type == "numeric_range":
                try:
                    numeric_value = float(value)
                except Exception:
                    return ValidationError(
                        error_type=self.ErrorCode.INVALID_TYPE,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "expected": "numeric string",
                            "rule": rule.get("name", "numeric_range"),
                        },
                    )
                min_val = rule.get("min")
                max_val = rule.get("max")
                if min_val is not None and numeric_value < float(min_val):
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "numeric_range"),
                            "message": rule.get("message", f"Value below minimum {min_val}"),
                            "min": min_val,
                        },
                    )
                if max_val is not None and numeric_value > float(max_val):
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "numeric_range"),
                            "message": rule.get("message", f"Value above maximum {max_val}"),
                            "max": max_val,
                        },
                    )
            elif rule_type == "must_contain":
                substr = rule.get("substring", "")
                if substr and substr not in value:
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "must_contain"),
                            "message": rule.get("message", f"Missing required substring '{substr}'"),
                            "substring": substr,
                        },
                    )
            elif rule_type == "must_not_contain":
                substr = rule.get("substring", "")
                if substr and substr in value:
                    return ValidationError(
                        error_type=self.ErrorCode.RULE_VIOLATION,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "must_not_contain"),
                            "message": rule.get("message", f"Contains disallowed substring '{substr}'"),
                            "substring": substr,
                        },
                    )
            elif rule_type == "regex_must_match":
                pattern = rule.get("pattern", "")
                if pattern and not re.match(pattern, value):
                    return ValidationError(
                        error_type=self.ErrorCode.INVALID_FORMAT,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "regex_must_match"),
                            "message": rule.get("message", "Value does not match required pattern"),
                            "pattern": pattern,
                        },
                    )
            elif rule_type == "regex_must_not_match":
                pattern = rule.get("pattern", "")
                if pattern and re.match(pattern, value):
                    return ValidationError(
                        error_type=self.ErrorCode.INVALID_FORMAT,
                        probability=1.0,
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": rule.get("name", "regex_must_not_match"),
                            "message": rule.get("message", "Value matches a forbidden pattern"),
                            "pattern": pattern,
                        },
                    )
        return None

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        # High-confidence validation: deterministic checks
        if pd.isna(value):
            return ValidationError(error_type=self.ErrorCode.MISSING_VALUE, probability=1.0, details={})
        if not isinstance(value, str):
            return ValidationError(
                error_type=self.ErrorCode.INVALID_TYPE,
                probability=1.0,
                details={"expected": "string", "actual": str(type(value))},
            )

        value_str = value
        # Apply validation rules (structural/business)
        rule_error = self._check_validation_rules(value_str)
        if rule_error:
            return rule_error

        # Apply format patterns (character/shape constraints)
        format_error = self._check_format_patterns(value_str)
        if format_error:
            return format_error

        # Enforce allowed values if provided
        if self.known_values:
            normalized_value = value_str.lower().strip()
            if normalized_value not in self.known_values:
                return ValidationError(
                    error_type=self.ErrorCode.VALUE_NOT_ALLOWED,
                    probability=1.0,
                    details={
                        "field": self.field_name,
                        "value": value_str,
                        "message": "Value not in allowed list",
                        "known_values_count": len(self.known_values),
                    },
                )

        # All checks passed
        return None