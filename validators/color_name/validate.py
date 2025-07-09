import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Optional

from validators.interfaces import ValidatorInterface

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
        EXTRA_SPACE = "EXTRA_SPACE"
        MERGED_WORDS = "MERGED_WORDS"
        RANDOM_TEXT_NOISE = "RANDOM_TEXT_NOISE"
        INVALID_FORMAT = "INVALID_FORMAT"

    def _validate_entry(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Contains the specific validation logic for a single data entry.
        This method MUST be implemented by the LLM.

        Args:
            value: The data from the DataFrame column to be validated.

        Returns:
            - None if the value is valid.
            - A dictionary with 'error_code' (str) and 'details' (dict) if the
              value is invalid.
        """
        # <<< LLM: BEGIN IMPLEMENTATION >>>
        
        # Check for missing values
        if pd.isna(value):
            return {"error_code": self.ErrorCode.MISSING_VALUE, "details": {}}
        
        # Check for invalid type
        if not isinstance(value, str):
            return {"error_code": self.ErrorCode.INVALID_TYPE, "details": {"expected": "string", "actual": str(type(value))}}
        
        # Trim the string for comparison but keep original for reporting
        original_value = value
        
        # Check for extra spaces (leading, trailing, or multiple consecutive spaces)
        if value != value.strip() or "  " in value:
            return {
                "error_code": self.ErrorCode.EXTRA_SPACE,
                "details": {"original": original_value, "suggested": " ".join(value.split())}
            }
        
        # Check for merged words (CamelCase with no spaces)
        # Look for patterns like "WordWord" where there should be a space
        import re
        if re.search(r'[a-z][A-Z]', value):
            # Simple heuristic to insert spaces before capital letters
            suggested = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
            return {
                "error_code": self.ErrorCode.MERGED_WORDS, 
                "details": {"original": original_value, "suggested": suggested}
            }
        
        # Check for random text noise (non-alphabetic characters except for spaces and slashes)
        # This is a simplified check that looks for digits and special characters
        if re.search(r'[0-9]|[^\w\s/]', value):
            # Remove digits and special characters except spaces and slashes
            clean_value = re.sub(r'[0-9]|[^\w\s/]', '', value)
            return {
                "error_code": self.ErrorCode.RANDOM_TEXT_NOISE,
                "details": {"original": original_value, "suggested": clean_value}
            }
        
        # Check for incorrect format using dashes instead of slashes
        if ' - ' in value:
            corrected = value.replace(' - ', '/')
            return {
                "error_code": self.ErrorCode.INVALID_FORMAT,
                "details": {"original": original_value, "suggested": corrected}
            }
        
        # If all checks pass, the value is valid
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