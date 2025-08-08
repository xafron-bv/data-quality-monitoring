import json
import os
from typing import Any, Dict, List

import pandas as pd

from common.exceptions import ConfigurationError, FileOperationError
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
        """
        self.validator_name = validator_name
        try:
            self.error_messages = self._load_error_messages(validator_name)
        except FileNotFoundError:
            # Fallback to generic messages for rule-based validator
            self.error_messages = self._get_generic_error_messages()
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in error messages file for validator '{validator_name}'",
                details={'validator_name': validator_name, 'json_error': str(e)}
            ) from e

    def _load_error_messages(self, validator_name):
        """Load error messages from the JSON file."""
        error_messages_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "validators", validator_name, "error_messages.json"
        )

        with open(error_messages_path, 'r') as f:
            return json.load(f)

    def _get_generic_error_messages(self) -> Dict[str, str]:
        return {
            "MISSING_VALUE": "Value is missing",
            "INVALID_TYPE": "Invalid type: expected {expected}, got {actual}",
            "INVALID_FORMAT": "Invalid {field} format: {message}",
            "VALUE_NOT_ALLOWED": "{value} is not an allowed {field} value",
            "RULE_VIOLATION": "Invalid {field}: {message}",
            "DEFAULT": "Unknown error with data: {error_data}",
        }

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
            if error_code in self.error_messages:
                message_template = self.error_messages[error_code]
            elif "DEFAULT" in self.error_messages:
                message_template = self.error_messages["DEFAULT"]
            else:
                message_template = "Unknown error with data: {error_data}"

            # Format the message with specific details from the error
            details = error.details.copy() if error.details else {}  # Create a copy to avoid modifying the original
            details['error_data'] = error.error_data  # Add original data for context
            details['probability'] = error.probability  # Add probability for potential use in message
            # Default field to validator name when available
            details.setdefault('field', self.validator_name)

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
                "probability": error.probability,     # Add probability to the report
                "error_code": error.error_type
            })
        return report
