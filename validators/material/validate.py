import pandas as pd
import re
from typing import List, Dict, Any, Optional

# The ValidatorInterface is expected to be in a file named 'interfaces.py'
from validators.interfaces import ValidatorInterface


class MaterialValidator(ValidatorInterface):
    """
    A high-speed, deterministic validator for material composition strings,
    refactored to use the ValidatorInterface template.
    """
    def __init__(self):
        """Initializes the validator and its tokenizer."""
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])')

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        normalized_s = re.sub(r'\s+', ' ', s.strip())
        return self.tokenizer_regex.findall(normalized_s)

    def _validate_entry(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Contains the specific validation logic for a single material string.
        This method checks for data quality, format, and composition rules.
        """
        # <<< LLM: BEGIN IMPLEMENTATION >>>

        # Rule: Handle missing, non-string, or empty values
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return {"error_code": "MISSING_VALUE", "details": {}}

        # Rule: Check for leading/trailing whitespace
        if value.strip() != value:
            return {"error_code": "EXTRANEOUS_WHITESPACE", "details": {}}

        # Rule: Check for line breaks
        if '\n' in value or '\r' in value:
            return {"error_code": "LINE_BREAK_FOUND", "details": {}}

        tokens = self._tokenize(value)

        if not tokens:
            return {"error_code": "EMPTY_PART", "details": {}}

        # Rule: Allow '.' for float values to prevent false positives.
        allowed_chars_regex = r'[^a-zA-Z0-9\s%\-\/\(\):\.]'
        invalid_chars = sorted(list(set([t for t in tokens if re.search(allowed_chars_regex, t)])))
        if invalid_chars:
            return {"error_code": "INVALID_CHARACTERS", "details": {"chars": invalid_chars}}

        # Rule: Detects ambiguous prefixes (e.g., "Blue Shell:")
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1].isalpha():
            return {"error_code": "AMBIGUOUS_PREFIX", "details": {"prefix": f"{tokens[0]} {tokens[1]}"}}

        # Rule: Detects prepended text (e.g., "Blue 100% Cotton")
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1] != ':':
            return {"error_code": "PREPENDED_TEXT_DETECTED", "details": {"text": tokens[0]}}

        # Rule: Check for invalid hyphen usage (e.g., "10%-Material")
        for i in range(len(tokens) - 1):
            if tokens[i].endswith('%') and tokens[i+1] == '-':
                return {"error_code": "INVALID_HYPHEN_DELIMITER", "details": {}}

        # Rule: Check for malformed tokens (e.g., "23uqtz%")
        for i in range(len(tokens) - 2):
            is_num = re.fullmatch(r'\d+(?:\.\d+)?', tokens[i])
            is_word = tokens[i+1].isalpha()
            is_percent = tokens[i+2] == '%'
            if is_num and is_word and is_percent:
                return {"error_code": "MALFORMED_TOKEN", "details": {"token": f"{tokens[i]}{tokens[i+1]}{tokens[i+2]}"}}
        
        # Rule: Check for numbers that are not followed by a percentage sign
        for token in tokens:
            if re.fullmatch(r'\d+(?:\.\d+)?', token):
                return {"error_code": "MISSING_PERCENTAGE_SIGN", "details": {"number": token}}

        # Rule: Check for duplicated material names (e.g., "100% Cotton Cotton")
        for i in range(len(tokens) - 1):
            if tokens[i].isalpha() and tokens[i] == tokens[i+1]:
                return {"error_code": "DUPLICATE_MATERIAL_NAME", "details": {"material": tokens[i]}}

        numbers = [float(t.replace('%','')) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?%', t)]
        if not numbers:
             return {"error_code": "MISSING_COMPOSITION", "details": {}}
        
        # Rule: Check for appended text (e.g., care instructions)
        last_percent_indices = [i for i, token in enumerate(tokens) if re.fullmatch(r'\d+(?:\.\d+)?%', token)]
        if last_percent_indices:
            last_percent_idx = last_percent_indices[-1]
            if len(tokens) > last_percent_idx + 2:
                for i in range(last_percent_idx + 2, len(tokens)):
                    if tokens[i].isalpha():
                        return {"error_code": "EXTRANEOUS_TEXT_APPENDED", "details": {"text": " ".join(tokens[i:])}}

        # Rule: Sum of percentages must be 100
        total_sum = sum(numbers)
        has_colon_delimiter = any(':' in t for t in tokens)
        
        if abs(total_sum - 100.0) > 1e-6:
            is_invalid_multi_part = has_colon_delimiter and (total_sum <= 100 or abs(total_sum % 100.0) > 1e-6)
            if not has_colon_delimiter or is_invalid_multi_part:
                 return {"error_code": "SUM_NOT_100", "details": {"sum": total_sum}}

        # If all checks pass, the value is valid.
        return None
        # <<< LLM: END IMPLEMENTATION >>>

    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[Dict[str, Any]]:
        """
        Validates a column and returns a list of structured errors.
        This method is a non-editable engine that runs the `_validate_entry` logic.
        """
        errors = []
        for index, row in df.iterrows():
            data = row[column_name]

            # The implemented logic in _validate_entry is called for every row.
            validation_error = self._validate_entry(data)

            # If the custom logic returned an error, format it as per the interface.
            if validation_error:
                errors.append({
                    "row_index": index,
                    "error_data": data,
                    **validation_error
                })
                
        return errors