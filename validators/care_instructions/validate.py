import pandas as pd
import re
from enum import Enum
from typing import List, Dict, Any, Optional

from validators.interfaces import ValidatorInterface

class Validator(ValidatorInterface):
    """
    A custom column validator to validate textile care instructions based on a
    pre-defined set of rules.

    This validator checks for a variety of issues including:
    - Missing or incorrect data types.
    - Formatting errors like extra whitespace, line breaks, and incorrect capitalization.
    - Presence of disallowed content like HTML tags, emojis, or special symbols.
    - Structural correctness of the instruction set, including delimiters and temperature format.
    """

    class ErrorCode(str, Enum):
        """Enumeration for validator error codes."""
        MISSING_VALUE = "MISSING_VALUE"
        INVALID_TYPE = "INVALID_TYPE"
        HAS_LEADING_OR_TRAILING_WHITESPACE = "HAS_LEADING_OR_TRAILING_WHITESPACE"
        CONTAINS_MULTIPLE_SPACES = "CONTAINS_MULTIPLE_SPACES"
        CONTAINS_LINE_BREAK = "CONTAINS_LINE_BREAK"
        CONTAINS_HTML = "CONTAINS_HTML"
        INCORRECT_CAPITALIZATION = "INCORRECT_CAPITALIZATION"
        CONTAINS_EMOJI = "CONTAINS_EMOJI"
        CONTAINS_DISALLOWED_SYMBOLS = "CONTAINS_DISALLOWED_SYMBOLS"
        INVALID_DELIMITER = "INVALID_DELIMITER"
        INVALID_TEMPERATURE_FORMAT = "INVALID_TEMPERATURE_FORMAT"
        UNKNOWN_INSTRUCTION = "UNKNOWN_INSTRUCTION"
        CONTAINS_PREPENDED_TEXT = "CONTAINS_PREPENDED_TEXT"
        CONTAINS_APPENDED_TEXT = "CONTAINS_APPENDED_TEXT"
        MISSING_INSTRUCTION = "MISSING_INSTRUCTION"

    def _validate_entry(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Contains the specific validation logic for a single data entry.
        """
        # <<< LLM: BEGIN IMPLEMENTATION >>>
        if pd.isna(value) or not str(value).strip():
            return {"error_code": self.ErrorCode.MISSING_VALUE, "details": {}}

        if not isinstance(value, str):
            return {"error_code": self.ErrorCode.INVALID_TYPE, "details": {"expected": "string"}}

        if value.strip() != value:
            return {"error_code": self.ErrorCode.HAS_LEADING_OR_TRAILING_WHITESPACE, "details": {}}
        
        # From this point, work with the stripped value
        value = value.strip()

        if not value.startswith("WASSEN OP MAX"):
            return {"error_code": self.ErrorCode.CONTAINS_PREPENDED_TEXT, "details": {}}

        if ' - ' in value and 'Machine wash cold' in value:
             return {"error_code": self.ErrorCode.CONTAINS_APPENDED_TEXT, "details": {}}

        if '  ' in value:
            return {"error_code": self.ErrorCode.CONTAINS_MULTIPLE_SPACES, "details": {}}

        if '\n' in value or '\r' in value:
            return {"error_code": self.ErrorCode.CONTAINS_LINE_BREAK, "details": {}}

        if re.search(r'<.*?>', value):
            return {"error_code": self.ErrorCode.CONTAINS_HTML, "details": {}}

        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF™®🧼]")
        if emoji_pattern.search(value):
            return {"error_code": self.ErrorCode.CONTAINS_EMOJI, "details": {}}

        if 'Ã' in value:
             return {"error_code": self.ErrorCode.CONTAINS_DISALLOWED_SYMBOLS, "details": {"symbol": "Ã"}}

        if ';' in value or ( ' - ' in value and 'NIET' in value):
            return {"error_code": self.ErrorCode.INVALID_DELIMITER, "details": {}}
        
        if '. .' in value or '..' in value:
            return {"error_code": self.ErrorCode.MISSING_INSTRUCTION, "details": {}}

        temp_pattern = r'WASSEN OP MAX (\d{2}°C)'
        if not re.search(temp_pattern, value):
             return {"error_code": self.ErrorCode.INVALID_TEMPERATURE_FORMAT, "details": {"value": value}}

        # Check for non-uppercase instructions
        instructions = value.split('. ')
        for instruction in instructions:
            # Find instruction that is not the temperature and is not fully uppercase
            if instruction and "WASSEN OP MAX" not in instruction and instruction != instruction.upper():
                return {"error_code": self.ErrorCode.INCORRECT_CAPITALIZATION, "details": {"instruction": instruction}}

        allowed_instructions = {
            'WASSEN OP MAX 30°C', 'WASSEN OP MAX 40°C', 'WAS IN VERGELIJKBARE KLEUREN', 
            'NIET BLEKEN', 'NIET IN DE DROGER', 'STRIJKEN OP LAGE TEMPERATUUR', 'NIET STRIJKEN', 'NIET STOMEN'
        }
        # Clean the value by removing trailing period for splitting
        cleaned_value = value.rstrip('.')
        parts = [part.strip() for part in cleaned_value.split('.') if part.strip()]
        for part in parts:
            if part not in allowed_instructions:
                 # Check again for valid temperature format before flagging as unknown
                if not re.fullmatch(r'WASSEN OP MAX \d{2}°C', part):
                    return {"error_code": self.ErrorCode.UNKNOWN_INSTRUCTION, "details": {"instruction": part}}

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