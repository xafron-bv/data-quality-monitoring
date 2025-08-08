import re
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator


class Validator(ValidatorInterface):
    """
    A high-speed, deterministic validator for material composition strings,
    refactored to use the ValidatorInterface template.
    """
    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        EXTRANEOUS_WHITESPACE = "EXTRANEOUS_WHITESPACE"
        LINE_BREAK_FOUND = "LINE_BREAK_FOUND"
        EMPTY_PART = "EMPTY_PART"
        INVALID_CHARACTERS = "INVALID_CHARACTERS"
        AMBIGUOUS_PREFIX = "AMBIGUOUS_PREFIX"
        PREPENDED_TEXT_DETECTED = "PREPENDED_TEXT_DETECTED"
        INVALID_HYPHEN_DELIMITER = "INVALID_HYPHEN_DELIMITER"
        MALFORMED_TOKEN = "MALFORMED_TOKEN"
        MISSING_PERCENTAGE_SIGN = "MISSING_PERCENTAGE_SIGN"
        DUPLICATE_MATERIAL_NAME = "DUPLICATE_MATERIAL_NAME"
        MISSING_COMPOSITION = "MISSING_COMPOSITION"
        EXTRANEOUS_TEXT_APPENDED = "EXTRANEOUS_TEXT_APPENDED"
        SUM_NOT_100 = "SUM_NOT_100"

    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="material")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        return self._engine._validate_entry(value)
