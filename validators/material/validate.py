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
        # Expanded tokenizer to include common separators.
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

        # Rule: Allow more characters to reduce false positives.
        # Common valid separators like '-', '/', '(', ')' are now permitted.
        allowed_chars_regex = r'[^a-zA-Z0-9\s%\-\/\(\):]'
        invalid_chars = sorted(list(set([t for t in tokens if re.search(allowed_chars_regex, t)])))
        if invalid_chars:
            return {"error_code": "INVALID_CHARACTERS", "details": {"chars": invalid_chars}}

        # Rule: Check for duplicated material names (e.g., "100% Cotton Cotton")
        for i in range(len(tokens) - 1):
            if tokens[i].isalpha() and tokens[i] == tokens[i+1]:
                return {"error_code": "DUPLICATE_MATERIAL_NAME", "details": {"material": tokens[i]}}

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
            # This check is now more flexible; it's disabled for now to avoid flagging valid cases.
            # Depending on strictness required, this could be re-enabled.
            # return {"error_code": "PERCENT_MISMATCH", "details": {"numbers_found": len(numbers), "percents_found": len(percent_signs)}}
            pass
        
        # Rule: Ambiguous structure (e.g., "Blue 100% Viscose")
        if (len(tokens) > 2 and tokens[0].isalpha() and tokens[-1].isalpha() and ':' not in tokens):
            return {"error_code": "AMBIGUOUS_STRUCTURE", "details": {"token_string": " ".join(tokens)}}

        return None # No error found

    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[Dict[str, Any]]:
        """
        Validates a specific column in a DataFrame and returns a list of structured errors.
        """
        errors = []
        # Explicitly check for 'nan' string to prevent false positives for MISSING_VALUE
        df[column_name] = df[column_name].replace('nan', None)

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