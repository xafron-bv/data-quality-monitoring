import pandas as pd
import json
import os
from typing import List, Dict, Any
from validators.interfaces import ReporterInterface

class MaterialReporter(ReporterInterface):
    """
    Implements the ReporterInterface to translate structured material validation
    errors into human-readable messages.
    """

    def __init__(self):
        """Initialize the reporter and load error messages from the JSON file."""
        # Load error messages from the JSON file
        error_messages_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "validators", "material", "error_messages.json"
        )
        with open(error_messages_path, 'r') as f:
            self.ERROR_MESSAGES = json.load(f)

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
