import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from validators.interfaces import ValidatorInterface
from validators.validation_error import ValidationError

class Validator(ValidatorInterface):
    """
    A completely empty template for a custom column validator.

    Instructions for LLM:
    1. Define your own error codes. Using a str-based Enum is recommended but not required.
    2. Implement ALL validation logic inside the `_validate_entry` method.
       This includes handling missing data (NaN/None), incorrect types, and empty values.
    3. The `_validate_entry` method must return `None` for valid data, or a dictionary
       containing 'error_type', 'confidence', and optionally 'details'.
    4. Do NOT modify the `bulk_validate` method.
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        # Define your error codes here
        # Example: MISSING_VALUE = "MISSING_VALUE"
        # Example: INVALID_TYPE = "INVALID_TYPE"
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"

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
        # <<< LLM: BEGIN IMPLEMENTATION >>>
        #
        # Implement validation logic here, using the ErrorCode enum.
        #
        # Example for a "user_age" column:
        #
        # if pd.isna(value):
        #     return ValidationError(
        #         error_type=self.ErrorCode.MISSING_VALUE,
        #         confidence=1.0,  # 100% confident it's missing
        #         details={}
        #     )
        #
        # if not isinstance(value, (int, float)):
        #     return ValidationError(
        #         error_type=self.ErrorCode.INVALID_TYPE,
        #         confidence=1.0,  # 100% confident it's the wrong type
        #         details={"expected": "integer"}
        #     )
        #
        # if not (18 <= value <= 120):
        #     return ValidationError(
        #         error_type=self.ErrorCode.AGE_OUT_OF_RANGE,
        #         confidence=0.95,  # 95% confident this is an error
        #         details={"age": value}
        #     )
        #

        # If all checks pass, the value is valid.
        return None

        # <<< LLM: END IMPLEMENTATION >>>


    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """
        Validates a column and returns a list of ValidationError objects.
        This method is a non-editable engine that runs the `_validate_entry` logic.
        """
        validation_errors = []
        for index, row in df.iterrows():
            data = row[column_name]

            # The implemented logic in _validate_entry is called for every row.
            validation_error = self._validate_entry(data)

            # If the custom logic returned an error, add context and add it to the list
            if validation_error:
                # Add row and column context to the validation error
                error_with_context = validation_error.with_context(
                    row_index=index,
                    column_name=column_name,
                    error_data=data
                )
                validation_errors.append(error_with_context)
                
        return validation_errors