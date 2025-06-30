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

        # Rule: Check for invalid characters (anything not alphanumeric, '%', or a space)
        invalid_chars = [t for t in tokens if re.search(r'[^a-zA-Z0-9\s%]', t)]
        if invalid_chars:
            return {"error_code": "INVALID_CHARACTERS", "details": {"chars": list(set(invalid_chars))}}

        # Rule: Sum of numbers must be 100
        numbers = [float(t.replace('%','')) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?%?', t)]
        if not numbers:
             return {"error_code": "MISSING_COMPOSITION", "details": {}}
        
        total_sum = sum(numbers)
        if abs(total_sum - 100.0) > 1e-6:
            return {"error_code": "SUM_NOT_100", "details": {"sum": total_sum}}

        # Rule: Number of '%' signs must match number of compositions
        percent_signs = [t for t in tokens if '%' in t]
        if len(percent_signs) != len(numbers):
            return {"error_code": "PERCENT_MISMATCH", "details": {"numbers_found": len(numbers), "percents_found": len(percent_signs)}}
        
        # Rule: Ambiguous structure (e.g., "Blue 100% Viscose")
        if (len(tokens) > 2 and tokens[0].isalpha() and tokens[-1].isalpha() and ':' not in tokens):
            return {"error_code": "AMBIGUOUS_STRUCTURE", "details": {"token_string": " ".join(tokens)}}

        return None # No error found

    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[Dict[str, Any]]:
        """
        Validates a specific column in a DataFrame and returns a list of structured errors.
        """
        errors = []
        for index, row in df.iterrows():
            material_string = row[column_name]
            
            if not isinstance(material_string, str) or not material_string.strip():
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
                 continue # Stop further validation on this error type

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
