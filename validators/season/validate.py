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
        INVALID_FORMAT = "INVALID_FORMAT"
        INVALID_YEAR = "INVALID_YEAR"
        INVALID_SEASON_NAME = "INVALID_SEASON_NAME"
        FUTURE_SEASON = "FUTURE_SEASON"

    def _validate_entry(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Contains the specific validation logic for a single data entry.
        This method validates season data which should be in format "YYYY Season"
        like "2022 Fall" or "2023 Summer".

        Args:
            value: The data from the DataFrame column to be validated.

        Returns:
            - None if the value is valid.
            - A dictionary with 'error_code' (str) and 'details' (dict) if the
              value is invalid.
        """
        # Check for missing values
        if pd.isna(value) or value == "":
            return {"error_code": self.ErrorCode.MISSING_VALUE, "details": {}}

        # Check for correct type
        if not isinstance(value, str):
            return {
                "error_code": self.ErrorCode.INVALID_TYPE, 
                "details": {"expected": "string", "received": str(type(value))}
            }

        # Check for valid format (YYYY Season)
        parts = value.strip().split(" ", 1)
        if len(parts) != 2:
            return {
                "error_code": self.ErrorCode.INVALID_FORMAT, 
                "details": {"value": value, "expected_format": "YYYY Season"}
            }

        year_str, season_name = parts

        # Check if year is numeric and 4 digits
        if not year_str.isdigit() or len(year_str) != 4:
            return {
                "error_code": self.ErrorCode.INVALID_YEAR, 
                "details": {"year": year_str}
            }

        # Convert year to integer
        year = int(year_str)
        
        # Check if year is reasonable (not too far in the past or future)
        current_year = 2025  # Based on context date
        if year < 1900 or year > current_year + 5:
            return {
                "error_code": self.ErrorCode.INVALID_YEAR, 
                "details": {"year": year, "min_year": 1900, "max_year": current_year + 5}
            }

        # Check if season name is valid
        valid_seasons = ["Spring", "Summer", "Fall", "Winter", "Resort", "Holiday", "Pre-Fall", "Pre-Spring"]
        if season_name not in valid_seasons:
            return {
                "error_code": self.ErrorCode.INVALID_SEASON_NAME, 
                "details": {"season": season_name, "valid_seasons": valid_seasons}
            }

        # If all checks pass, the value is valid
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