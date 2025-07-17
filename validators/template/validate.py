import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from validators.validator_interface import ValidatorInterface
from validators.validation_error import ValidationError

class Validator(ValidatorInterface):
    """
    A completely empty template for a custom column validator.

    Instructions for LLM:
    1. Define your own error codes. Using a str-based Enum is recommended but not required.
    2. Implement ALL validation logic inside the `_validate_entry` method.
       This includes handling missing data (NaN/None), incorrect types, and empty values.
    3. The `_validate_entry` method must return `None` for valid data, or a dictionary
       containing 'error_type', 'probability', and optionally 'details'.
    4. bulk_validate method is implemented in the ValidatorInterface and is responsible for
       applying the `_validate_entry` method to each entry in the DataFrame column. Do NOT rewrite it.
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
        #         probability=1.0,  # 100% confident it's missing
        #         details={}
        #     )
        #
        # if not isinstance(value, (int, float)):
        #     return ValidationError(
        #         error_type=self.ErrorCode.INVALID_TYPE,
        #         probability=1.0,  # 100% confident it's the wrong type
        #         details={"expected": "integer"}
        #     )
        #
        # if not (18 <= value <= 120):
        #     return ValidationError(
        #         error_type=self.ErrorCode.AGE_OUT_OF_RANGE,
        #         probability=0.95,  # 95% confident this is an error
        #         details={"age": value}
        #     )
        #

        # If all checks pass, the value is valid.
        return None

        # <<< LLM: END IMPLEMENTATION >>>