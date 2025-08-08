import re
from enum import Enum
from typing import Any, Optional

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator


class Validator(ValidatorInterface):
    """
    A completely empty template for a custom column validator.

    Instructions for LLM:
    1. Define your own error codes. Using a str-based Enum is recommended but not required.
    2. Implement ALL validation logic inside the `_validate_entry` method.
       This includes handling missing data (NaN/None), incorrect types, and empty values.
    3. The `_validate_entry` method must return `None` for valid data, or a dictionary
       containing 'error_code' and 'details' for invalid data.
    4. Do NOT modify the `bulk_validate` method.
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        EMPTY_CATEGORY = "EMPTY_CATEGORY"
        WHITESPACE_ERROR = "WHITESPACE_ERROR"
        RANDOM_NOISE = "RANDOM_NOISE"
        SPECIAL_CHARACTERS = "SPECIAL_CHARACTERS"
        HTML_TAGS = "HTML_TAGS"

    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="category")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """
        Contains the specific validation logic for a single data entry.
        This method MUST be implemented by the LLM.

        Args:
            value: The data from the DataFrame column to be validated.

        Returns:
            - None if the value is valid.
            - A ValidationError instance if the value is invalid.
        """
        return self._engine._validate_entry(value)
