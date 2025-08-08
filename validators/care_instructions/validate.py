import re
from enum import Enum
from typing import Any, Optional

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator


class Validator(ValidatorInterface):
    """
    A custom column validator to validate textile care instructions based on a
    pre-defined set of rules.

    This validator checks for a variety of issues including:
    - Missing or incorrect data types.
    - Formatting errors like extra whitespace, line breaks, and incorrect capitalization.
    - Presence of disallowed content like HTML tags, emojis, or special symbols.
    - Structural correctness of the instruction set, including delimiters and temperature format.
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        HAS_LEADING_OR_TRAILING_WHITESPACE = "HAS_LEADING_OR_TRAILING_WHITESPACE"
        CONTAINS_MULTIPLE_SPACES = "CONTAINS_MULTIPLE_SPACES"
        CONTAINS_LINE_BREAK = "CONTAINS_LINE_BREAK"
        CONTAINS_HTML = "CONTAINS_HTML"
        INCORRECT_CAPITALIZATION = "INCORRECT_CAPITALIZATION"
        CONTAINS_EMOJI = "CONTAINS_EMOJI"
        CONTAINS_DISALLOWED_SYMBOLS = "CONTAINS_DISALLOWED_SYMBOLS"
        INVALID_DELIMITER = "INVALID_DELIMITER"
        INVALID_TEMPERATURE_FORMAT = "INVALID_TEMPERATURE_FORMAT"
        UNKNOWN_INSTRUCTION = "UNKNOWN_INSTRUCTION"
        CONTAINS_PREPENDED_TEXT = "CONTAINS_PREPENDED_TEXT"
        CONTAINS_APPENDED_TEXT = "CONTAINS_APPENDED_TEXT"
        MISSING_INSTRUCTION = "MISSING_INSTRUCTION"

    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="care_instructions")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        return self._engine._validate_entry(value)
