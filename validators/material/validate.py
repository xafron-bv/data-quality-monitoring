import pandas as pd
import re
from typing import List, Dict, Any, Optional
from validators.interfaces import ValidatorInterface

class MaterialValidator(ValidatorInterface):
    """
    A high-speed, deterministic validator for material composition strings that
    now implements the ValidatorInterface and produces structured error codes.
    """
    def __init__(self):
        """Initializes the validator."""
        # Tokenizer remains the same, logic is enhanced in the validation steps.
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])')

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        normalized_s = re.sub(r'\s+', ' ', s.strip())
        return self.tokenizer_regex.findall(normalized_s)

    def _validate_part(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        """
        Checks a token sequence for validity, returning a specific error if one is found.
        """
        if not tokens:
            return {"error_code": "EMPTY_PART", "details": {}}

        # Rule: Allow '.' for float values to prevent false positives.
        allowed_chars_regex = r'[^a-zA-Z0-9\s%\-\/\(\):\.]'
        invalid_chars = sorted(list(set([t for t in tokens if re.search(allowed_chars_regex, t)])))
        if invalid_chars:
            return {"error_code": "INVALID_CHARACTERS", "details": {"chars": invalid_chars}}

        # Rule: Detects ambiguous prefixes like "Blue Shell:..." by checking for consecutive alphabetic tokens at the start.
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1].isalpha():
            return {"error_code": "AMBIGUOUS_PREFIX", "details": {"prefix": f"{tokens[0]} {tokens[1]}"}}

        # New Rule: Detects prepended text (e.g., "Blue 100%...") by checking for an initial alphabetic token
        # that is not part of a "Part:" declaration. This addresses the false negatives from the report.
        if len(tokens) > 1 and tokens[0].isalpha() and tokens[1] != ':':
            return {"error_code": "PREPENDED_TEXT_DETECTED", "details": {"text": tokens[0]}}

        # Rule: Check for invalid hyphen usage (e.g., "10%-Material")
        for i in range(len(tokens) - 1):
            if tokens[i].endswith('%') and tokens[i+1] == '-':
                return {"error_code": "INVALID_HYPHEN_DELIMITER", "details": {}}

        # Rule: Check for malformed tokens (e.g., "23uqtz%").
        # This identifies a number, followed by a word, followed by a percent sign.
        for i in range(len(tokens) - 2):
            is_num = re.fullmatch(r'\d+(?:\.\d+)?', tokens[i])
            is_word = tokens[i+1].isalpha()
            is_percent = tokens[i+2] == '%'
            if is_num and is_word and is_percent:
                return {"error_code": "MALFORMED_TOKEN", "details": {"token": f"{tokens[i]}{tokens[i+1]}{tokens[i+2]}"}}
        
        # Rule: Check for numbers that are not followed by a percentage sign.
        for token in tokens:
            if re.fullmatch(r'\d+(?:\.\d+)?', token):
                return {"error_code": "MISSING_PERCENTAGE_SIGN", "details": {"number": token}}

        # Rule: Check for duplicated material names (e.g., "100% Cotton Cotton")
        for i in range(len(tokens) - 1):
            if tokens[i].isalpha() and tokens[i] == tokens[i+1]:
                return {"error_code": "DUPLICATE_MATERIAL_NAME", "details": {"material": tokens[i]}}

        # Extract numbers ONLY from tokens with a '%' sign.
        numbers = [float(t.replace('%','')) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?%', t)]
        if not numbers:
             return {"error_code": "MISSING_COMPOSITION", "details": {}}
        
        # Rule: Check for appended text (e.g., care instructions) after the material composition.
        last_percent_indices = [i for i, token in enumerate(tokens) if re.fullmatch(r'\d+(?:\.\d+)?%', token)]
        if last_percent_indices:
            last_percent_idx = last_percent_indices[-1]
            # Check for any alphabetic tokens after the last material declaration.
            # We search from two positions after the last percentage sign to account for the material name.
            if len(tokens) > last_percent_idx + 2:
                for i in range(last_percent_idx + 2, len(tokens)):
                    if tokens[i].isalpha():
                        return {"error_code": "EXTRANEOUS_TEXT_APPENDED", "details": {"text": " ".join(tokens[i:])}}

        # Rule: Sum of numbers must be 100, or a multiple of 100 for multi-part materials.
        total_sum = sum(numbers)
        has_colon_delimiter = any(':' in t for t in tokens)
        
        if abs(total_sum - 100.0) > 1e-6:
            is_invalid_multi_part = has_colon_delimiter and (total_sum <= 100 or abs(total_sum % 100.0) > 1e-6)
            if not has_colon_delimiter or is_invalid_multi_part:
                 return {"error_code": "SUM_NOT_100", "details": {"sum": total_sum}}

        return None # No error found

    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[Dict[str, Any]]:
        """
        Validates a specific column in a DataFrame and returns a list of structured errors.
        """
        errors = []
        # Do not treat 'nan' string as a missing value upfront. Let the validator check it.
        
        for index, row in df.iterrows():
            material_string = row[column_name]
            
            # Refined check for missing or non-string values
            if pd.isna(material_string) or not isinstance(material_string, str) or not material_string.strip():
                errors.append({
                    "row_index": index,
                    "error_data": material_string,
                    "error_code": "MISSING_VALUE",
                    "details": {}
                })
                continue
            
            # Rule: Check for leading/trailing whitespace
            if material_string.strip() != material_string:
                 errors.append({
                    "row_index": index,
                    "error_data": material_string,
                    "error_code": "EXTRANEOUS_WHITESPACE",
                    "details": {}
                })
                 continue

            # Rule: Check for line breaks
            if '\n' in material_string or '\r' in material_string:
                errors.append({
                    "row_index": index,
                    "error_data": material_string,
                    "error_code": "LINE_BREAK_FOUND",
                    "details": {}
                })
                continue

            tokens = self._tokenize(material_string)
            validation_error = self._validate_part(tokens)

            if validation_error:
                errors.append({
                    "row_index": index,
                    "error_data": material_string,
                    **validation_error
                })
        return errors