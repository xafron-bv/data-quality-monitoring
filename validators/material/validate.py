import pandas as pd
import re
import pandas as pd
import re
from enum import Enum
from typing import List, Dict, Any, Optional

from validators.validator_interface import ValidatorInterface
from validators.validation_error import ValidationError


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
    
    def __init__(self):
        """Initializes the validator and its tokenizer."""
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])')

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        normalized_s = re.sub(r'\s+', ' ', s.strip())
        return self.tokenizer_regex.findall(normalized_s)

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """
        Contains the specific validation logic for a single material string.
        This method checks for data quality, format, and composition rules.
        """
        # <<< LLM: BEGIN IMPLEMENTATION >>>

        # Rule: Handle missing, non-string, or empty values
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return {"error_code": self.ErrorCode.MISSING_VALUE, "details": {}}

        # Rule: Check for leading/trailing whitespace
        if value.strip() != value:
            return {"error_code": self.ErrorCode.EXTRANEOUS_WHITESPACE, "details": {}}

        # Rule: Check for line breaks
        if '\n' in value or '\r' in value:
            return {"error_code": self.ErrorCode.LINE_BREAK_FOUND, "details": {}}

        tokens = self._tokenize(value)

        if not tokens:
            return {"error_code": self.ErrorCode.EMPTY_PART, "details": {}}

        # Rule: Allow '.' for float values to prevent false positives.
        allowed_chars_regex = r'[^a-zA-Z0-9\s%\-\/\(\):\.]'
        invalid_chars = sorted(list(set([t for t in tokens if re.search(allowed_chars_regex, t)])))
        if invalid_chars:
            return {"error_code": self.ErrorCode.INVALID_CHARACTERS, "details": {"chars": invalid_chars}}

        # Rule: Detects ambiguous prefixes (e.g., "Blue Shell:")
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1].isalpha():
            return {"error_code": self.ErrorCode.AMBIGUOUS_PREFIX, "details": {"prefix": f"{tokens[0]} {tokens[1]}"}}

        # Rule: Detects prepended text (e.g., "Blue 100% Cotton")
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1] != ':':
            return {"error_code": self.ErrorCode.PREPENDED_TEXT_DETECTED, "details": {"text": tokens[0]}}

        # Rule: Check for invalid hyphen usage (e.g., "10%-Material")
        for i in range(len(tokens) - 1):
            if tokens[i].endswith('%') and tokens[i+1] == '-':
                return {"error_code": self.ErrorCode.INVALID_HYPHEN_DELIMITER, "details": {}}

        # Rule: Check for malformed tokens (e.g., "23uqtz%")
        for i in range(len(tokens) - 2):
            is_num = re.fullmatch(r'\d+(?:\.\d+)?', tokens[i])
            is_word = tokens[i+1].isalpha()
            is_percent = tokens[i+2] == '%'
            if is_num and is_word and is_percent:
                return {"error_code": self.ErrorCode.MALFORMED_TOKEN, "details": {"token": f"{tokens[i]}{tokens[i+1]}{tokens[i+2]}"}}
        
        # Rule: Check for numbers that are not followed by a percentage sign
        for token in tokens:
            if re.fullmatch(r'\d+(?:\.\d+)?', token):
                return {"error_code": self.ErrorCode.MISSING_PERCENTAGE_SIGN, "details": {"number": token}}

        # Rule: Check for duplicated material names (e.g., "100% Cotton Cotton")
        for i in range(len(tokens) - 1):
            if tokens[i].isalpha() and tokens[i] == tokens[i+1]:
                return {"error_code": self.ErrorCode.DUPLICATE_MATERIAL_NAME, "details": {"material": tokens[i]}}

        numbers = [float(t.replace('%','')) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?%', t)]
        if not numbers:
             return {"error_code": self.ErrorCode.MISSING_COMPOSITION, "details": {}}
        
        # Rule: Check for appended text (e.g., care instructions)
        last_percent_indices = [i for i, token in enumerate(tokens) if re.fullmatch(r'\d+(?:\.\d+)?%', token)]
        if last_percent_indices:
            last_percent_idx = last_percent_indices[-1]
            if len(tokens) > last_percent_idx + 2:
                for i in range(last_percent_idx + 2, len(tokens)):
                    if tokens[i].isalpha():
                        return {"error_code": self.ErrorCode.EXTRANEOUS_TEXT_APPENDED, "details": {"text": " ".join(tokens[i:])}}

        # Rule: Sum of percentages must be 100
        total_sum = sum(numbers)
        has_colon_delimiter = any(':' in t for t in tokens)
        
        if abs(total_sum - 100.0) > 1e-6:
            is_invalid_multi_part = has_colon_delimiter and (total_sum <= 100 or abs(total_sum % 100.0) > 1e-6)
            if not has_colon_delimiter or is_invalid_multi_part:
                 return {"error_code": self.ErrorCode.SUM_NOT_100, "details": {"sum": total_sum}}

        # If all checks pass, the value is valid.
        return None
        # <<< LLM: END IMPLEMENTATION >>>