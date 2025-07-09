import pandas as pd
import re
from enum import Enum
from typing import List, Dict, Any, Optional

from validators.interfaces import ValidatorInterface

class Validator(ValidatorInterface):
    """
    A validator for clothing size data.
    
    This validator checks for common errors in size values including:
    - Invalid characters or random noise in size values
    - Invalid size prefixes 
    - Fractional size errors
    - Leading/trailing spaces
    - Appended size suffixes
    - Decimal in integer size
    - Wrong size delimiters
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        RANDOM_NOISE = "RANDOM_NOISE"
        INVALID_PREFIX = "INVALID_PREFIX"
        FRACTIONAL_SIZE = "FRACTIONAL_SIZE"
        LEADING_TRAILING_SPACE = "LEADING_TRAILING_SPACE"
        APPENDED_SUFFIX = "APPENDED_SUFFIX"
        DECIMAL_IN_INTEGER = "DECIMAL_IN_INTEGER"
        WRONG_DELIMITER = "WRONG_DELIMITER"

    def _validate_entry(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Validates a size value entry for common errors.

        Args:
            value: The data from the DataFrame column to be validated.

        Returns:
            - None if the value is valid.
            - A dictionary with 'error_code' (str) and 'details' (dict) if the
              value is invalid.
        """
        # Check for missing values
        if pd.isna(value) or value == '':
            return {"error_code": self.ErrorCode.MISSING_VALUE, "details": {}}
        
        # Ensure value is a string for proper validation
        if not isinstance(value, str):
            # Convert numerical values to strings for further validation
            if isinstance(value, (int, float)):
                value = str(value)
            else:
                return {"error_code": self.ErrorCode.INVALID_TYPE, 
                        "details": {"expected": "string or numeric", "received": str(type(value))}}
        
        # Check for leading or trailing spaces
        if value != value.strip():
            return {"error_code": self.ErrorCode.LEADING_TRAILING_SPACE, 
                    "details": {"original": value, "stripped": value.strip()}}
        
        # Check for invalid prefixes like "SIZE:"
        if re.match(r'^SIZE:\s', value, re.IGNORECASE):
            return {"error_code": self.ErrorCode.INVALID_PREFIX, 
                    "details": {"prefix": value.split()[0]}}
        
        # Check for appended suffixes like "(Perfect Fit)"
        if re.search(r'\(.+\)$', value):
            match = re.search(r'(.*?)(\(.+\))$', value)
            return {"error_code": self.ErrorCode.APPENDED_SUFFIX, 
                    "details": {"size_part": match.group(1).strip(), "suffix": match.group(2)}}
        
        # Check for incorrect fractional notation with '$'
        if re.match(r'^\$\d+\s\d+/\d+$', value):
            return {"error_code": self.ErrorCode.FRACTIONAL_SIZE, 
                    "details": {"fractional_size": value}}
        
        # Check for decimal in integer size with '$'
        if re.match(r'^\$\d+\.\d+$', value):
            return {"error_code": self.ErrorCode.DECIMAL_IN_INTEGER, 
                    "details": {"decimal_size": value}}
        
        # Check for wrong delimiters (e.g., "M-L" instead of "M/L")
        if re.match(r'^[XSMLxsml]+-[XSMLxsml]+$', value):
            return {"error_code": self.ErrorCode.WRONG_DELIMITER, 
                    "details": {"delimiter_value": value}}
        
        # Check for random noise/invalid characters in size
        # Valid sizes should be either standard letter sizes or numeric sizes
        valid_patterns = [
            r'^[XSMLxsml]+$',                 # Letter sizes: S, M, L, XL, etc.
            r'^[0-9]+$',                      # Numeric sizes: 36, 38, 40, etc.
            r'^[0-9]+/[0-9]+$',               # Fractional sizes: 6/8
            r'^[XSMLxsml]+/[XSMLxsml]+$',     # Size ranges: S/M
            r'^[0-9]+\.[0-9]+$'               # Decimal sizes: 7.5
        ]
        
        if not any(re.match(pattern, value) for pattern in valid_patterns):
            # If it doesn't match any of our valid patterns, it might contain random noise
            return {"error_code": self.ErrorCode.RANDOM_NOISE, 
                    "details": {"invalid_size": value}}
        
        # If all checks pass, the value is valid.
        return None


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