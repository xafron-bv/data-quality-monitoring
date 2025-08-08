"""
JSON-based reporter that works with the new validation rules structure.
"""

import json
import os
from typing import Any, Dict, List

from validators.reporter_interface import ReporterInterface
from validators.validation_error import ValidationError
from common.exceptions import FileOperationError


class JSONReporter(ReporterInterface):
    """
    Reporter that loads error messages from the new JSON validation rules files.
    """
    
    def __init__(self, field_name: str):
        """
        Initialize the reporter and load error messages from the JSON rules file.
        
        Args:
            field_name: The name of the field to load error messages for.
        """
        self.field_name = field_name
        try:
            self.error_messages = self._load_error_messages(field_name)
        except FileNotFoundError:
            raise FileOperationError(
                f"Validation rules file not found for field '{field_name}'",
                details={'field_name': field_name}
            )
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in validation rules file for field '{field_name}'",
                details={'field_name': field_name, 'json_error': str(e)}
            ) from e
    
    def _load_error_messages(self, field_name: str) -> Dict[str, str]:
        """Load error messages from the JSON rules file."""
        rules_path = os.path.join(
            os.path.dirname(__file__),
            'rules',
            f'{field_name}.json'
        )
        
        with open(rules_path, 'r') as f:
            rules = json.load(f)
            return rules.get('error_messages', {})
    
    def format_error_message(self, error: ValidationError) -> str:
        """
        Format an error message using the template and error details.
        
        Args:
            error: The ValidationError instance
            
        Returns:
            Formatted error message string
        """
        template = self.error_messages.get(error.error_type, f"Validation error: {error.error_type}")
        
        # Format the message with the error details
        try:
            return template.format(**error.details)
        except KeyError as e:
            # If a placeholder is missing, return a generic message
            return f"{template} (Details: {error.details})"
        except Exception:
            # Fallback for any other formatting issues
            return f"Validation error: {error.error_type} - {error.details}"
    
    def generate_summary(self, errors: List[ValidationError]) -> Dict[str, int]:
        """
        Generate a summary of validation errors by type.
        
        Args:
            errors: List of ValidationError instances
            
        Returns:
            Dictionary mapping error types to counts
        """
        summary = {}
        for error in errors:
            error_type = error.error_type
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary
    
    def generate_report(self, errors: List[ValidationError], df=None) -> List[Dict[str, Any]]:
        """
        Generate a report of validation errors compatible with the existing system.
        
        Args:
            errors: List of ValidationError instances
            df: The original dataframe (unused but kept for compatibility)
            
        Returns:
            List of error report dictionaries
        """
        report = []
        for error in errors:
            # Format the error message
            display_message = self.format_error_message(error)
            
            report.append({
                "row_index": error.row_index,
                "error_data": error.error_data,
                "display_message": display_message,
                "column_name": error.column_name,
                "probability": error.probability,
                "error_code": error.error_type
            })
        
        return report