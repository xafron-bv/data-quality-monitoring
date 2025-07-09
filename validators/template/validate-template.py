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
        # Define your error codes here
        # Example: MISSING_VALUE = "MISSING_VALUE"
        # Example: INVALID_TYPE = "INVALID_TYPE"
        pass

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
        #
        # Define error codes and all validation logic here.
        #
        # Example for a "user_age" column:
        #
        # if pd.isna(value):
        #     return {"error_code": "MISSING_AGE", "details": {}}
        #
        # if not isinstance(value, (int, float)):
        #     return {"error_code": "INVALID_TYPE", "details": {"expected": "integer"}}
        #
        # if not (18 <= value <= 120):
        #     return {"error_code": "AGE_OUT_OF_RANGE", "details": {"age": value}}
        #

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