import pandas as pd
import json
import os
from typing import List, Dict, Any
from validators.reporter_interface import ReporterInterface
from validators.validation_error import ValidationError

class Reporter(ReporterInterface):
    """
    Implements the ReporterInterface to translate structured validation
    errors into human-readable messages for any validator.
    """

    def __init__(self, validator_name):
        """
        Initialize the reporter and load error messages from the JSON file.
        
        Args:
            validator_name (str): The name of the validator to load error messages for.
                                 This is mandatory and must be provided.
        """
        # Load error messages from the JSON file
        if validator_name is None or validator_name.strip() == "":
            print("Error: validator_name is required for Reporter initialization")
            import sys
            sys.exit(1)
            
        error_messages_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "validators", validator_name, "error_messages.json"
        )
        try:
            with open(error_messages_path, 'r') as f:
                self.ERROR_MESSAGES = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find error messages file at {error_messages_path}")
            print(f"Please ensure that validator '{validator_name}' exists and has an error_messages.json file.")
            import sys
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The error messages file at {error_messages_path} contains invalid JSON.")
            import sys
            sys.exit(1)

    def generate_report(self, validation_errors: List[ValidationError], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of validation errors from any validator.
        
        Args:
            validation_errors: List of ValidationError objects from a validator
            original_df: Original dataframe that was validated
            
        Returns:
            List of reports with human-readable error messages
        """
        report = []
        for error in validation_errors:
            error_code = error.error_type
            
            # First try to get the template for the specific error code
            # If not found, try to use DEFAULT, and if that's not available, use a generic message
            if error_code in self.ERROR_MESSAGES:
                message_template = self.ERROR_MESSAGES[error_code]
            elif "DEFAULT" in self.ERROR_MESSAGES:
                message_template = self.ERROR_MESSAGES["DEFAULT"]
            else:
                message_template = "Unknown error with data: {error_data}"
            
            # Format the message with specific details from the error
            details = error.details.copy() if error.details else {}  # Create a copy to avoid modifying the original
            details['error_data'] = error.error_data  # Add original data for context
            details['confidence'] = error.confidence  # Add confidence for potential use in message
            
            try:
                display_message = message_template.format(**details)
            except KeyError as e:
                # Handle missing format keys gracefully
                display_message = f"Error formatting message: {message_template} (missing {e})"
            except Exception as e:
                display_message = f"Error formatting message: {str(e)}"

            report.append({
                "row_index": error.row_index,
                "error_data": error.error_data,
                "display_message": display_message,
                "column_name": error.column_name,  # Add column name to the report
                "confidence": error.confidence     # Add confidence to the report
            })
        return report
