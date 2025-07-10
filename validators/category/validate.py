import pandas as pd
import re
from enum import Enum
from typing import List, Dict, Any, Optional

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
       containing 'error_code' and 'details' for invalid data.
    4. Do NOT modify the `bulk_validate` method.
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        EMPTY_CATEGORY = "EMPTY_CATEGORY"
        WHITESPACE_ERROR = "WHITESPACE_ERROR"
        RANDOM_NOISE = "RANDOM_NOISE"
        SPECIAL_CHARACTERS = "SPECIAL_CHARACTERS"
        HTML_TAGS = "HTML_TAGS"

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
        import re

        # Check for missing value
        if pd.isna(value):
            return ValidationError(
                error_type=self.ErrorCode.MISSING_VALUE,
                confidence=1.0,
                details={}
            )
        
        # Check for invalid type
        if not isinstance(value, str):
            return ValidationError(
                error_type=self.ErrorCode.INVALID_TYPE,
                confidence=1.0,
                details={"expected": "string", "actual": str(type(value))}
            )
        
        # Check for empty category
        if value == "":
            return ValidationError(
                error_type=self.ErrorCode.EMPTY_CATEGORY,
                confidence=1.0,
                details={}
            )
        
        # Check for whitespace errors (leading/trailing spaces)
        if value.strip() != value:
            return ValidationError(
                error_type=self.ErrorCode.WHITESPACE_ERROR,
                confidence=0.95,
                details={"original": value, "stripped": value.strip()}
            )
        
        # Check for HTML tags
        if re.search(r'<[^>]+>', value):
            return ValidationError(
                error_type=self.ErrorCode.HTML_TAGS,
                confidence=0.98,
                details={"category": value}
            )
        
        # Check for special characters (excluding alphanumeric, spaces, hyphens, and underscores)
        if re.search(r'[^\w\s\-]', value):
            return ValidationError(
                error_type=self.ErrorCode.SPECIAL_CHARACTERS,
                confidence=0.9,
                details={"category": value}
            )
        
        # Check for random noise (characters that don't form valid words)
        # More comprehensive check for random noise - detects:
        # 1. Mixed case with numbers embedded within words
        # 2. Random character sequences like alternating upper/lowercase
        if re.search(r'[A-Za-z]+\d+[A-Za-z]+', value) or \
           re.search(r'([A-Z][a-z]){2,}', value) or \
           re.search(r'([a-z][A-Z]){2,}', value) or \
           re.search(r'[A-Za-z]\d[A-Za-z]', value):
            return ValidationError(
                error_type=self.ErrorCode.RANDOM_NOISE,
                confidence=0.85,
                details={"category": value}
            )

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