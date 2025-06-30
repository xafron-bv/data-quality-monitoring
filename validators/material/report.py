import pandas as pd
from typing import List, Dict, Any
from validators.interfaces import ReporterInterface

class MaterialReporter(ReporterInterface):
    """
    Implements the ReporterInterface to translate structured material validation
    errors into human-readable messages.
    """

    ERROR_MESSAGES = {
        "MISSING_VALUE": "The material field is empty or missing.",
        "EXTRANEOUS_WHITESPACE": "The material string has leading or trailing spaces.",
        "LINE_BREAK_FOUND": "The material string contains a line break, which is not allowed.",
        "INVALID_CHARACTERS": "The material string contains invalid characters: {chars}. Only letters, numbers, and '%' are permitted.",
        "SUM_NOT_100": "The composition percentages sum to {sum}, but should sum to 100.",
        "MISSING_COMPOSITION": "The material string lists a material name but is missing a numeric percentage.",
        "PERCENT_MISMATCH": "Mismatch between numbers and percent signs. Found {numbers_found} numbers and {percents_found} '%' signs.",
        "AMBIGUOUS_STRUCTURE": "The structure is ambiguous. It starts and ends with a word without a clear delimiter (e.g., 'Color 100% Material').",
        "DEFAULT": "The material string '{error_data}' is invalid for an unknown reason."
    }

    def generate_report(self, validation_errors: List[Dict[str, Any]], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of material validation errors.
        """
        report = []
        for error in validation_errors:
            error_code = error.get("error_code", "DEFAULT")
            message_template = self.ERROR_MESSAGES.get(error_code, self.ERROR_MESSAGES["DEFAULT"])
            
            # Format the message with specific details from the error
            details = error.get("details", {})
            details['error_data'] = error.get('error_data') # Add original data for context
            display_message = message_template.format(**details)

            report.append({
                "row_index": error["row_index"],
                "error_data": error["error_data"],
                "display_message": display_message
            })
        return report
