import pandas as pd
import json
import os
import sys
from typing import List, Dict, Any
from anomaly_detectors.reporter_interface import AnomalyReporterInterface
from anomaly_detectors.anomaly_error import AnomalyError

# Add the parent directory to the path to import our exception module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from common.exceptions import ConfigurationError, FileOperationError

class AnomalyReporter(AnomalyReporterInterface):
    """
    Generic anomaly reporter that works with the unified pattern-based detector.
    Uses generic error message templates instead of field-specific message files.
    """

    def __init__(self, detector_name):
        """
        Initialize the reporter with generic error message templates.
        
        Args:
            detector_name (str): The name of the detector (for context).
        """
        self.detector_name = detector_name
        self.error_messages = self._get_generic_error_messages()

    def _get_generic_error_messages(self):
        """Return generic error message templates for pattern-based detection."""
        return {
            "INVALID_VALUE": "Invalid value detected: {details}",
            "UNKNOWN_VALUE": "Unknown value '{value}' not found in valid {field} values",
            "INVALID_FORMAT": "Invalid format in {field}: {message}",
            "SUSPICIOUS_PATTERN": "Suspicious pattern detected in {field}: {details}",
            "DOMAIN_VIOLATION": "Domain violation in {field}: {details}"
        }

    def _format_error_message(self, anomaly: AnomalyError) -> str:
        """
        Format an error message based on the anomaly type and details.
        
        Args:
            anomaly: The AnomalyError object to format
            
        Returns:
            Formatted human-readable error message
        """
        error_type = anomaly.anomaly_type
        details = anomaly.details or {}
        
        # Get base template
        template = self.error_messages.get(error_type, f"Anomaly detected: {error_type}")
        
        # Format with available details
        try:
            if error_type == "UNKNOWN_VALUE":
                field = details.get('field', 'field')
                value = details.get('value', 'unknown')
                possible_matches = details.get('possible_matches', [])
                
                message = f"Unknown {field} value '{value}'"
                if possible_matches:
                    message += f". Did you mean: {', '.join(possible_matches[:3])}?"
                return message
                
            elif error_type == "INVALID_FORMAT":
                field = details.get('field', 'field')
                message = details.get('message', 'format issue')
                return f"Invalid {field} format: {message}"
                
            elif error_type == "INVALID_VALUE":
                field = details.get('field', 'field')
                message = details.get('message', 'invalid value')
                return f"Invalid {field}: {message}"
                
            elif error_type == "DOMAIN_VIOLATION":
                field = details.get('field', 'field')
                reason = details.get('reason', 'domain violation')
                return f"Domain violation in {field}: {reason}"
                
            else:
                # Generic fallback
                field = details.get('field', self.detector_name)
                return template.format(field=field, details=str(details), **details)
                
        except (KeyError, ValueError):
            # Fallback to simple message
            return f"{error_type}: {str(details)}"

    def generate_report(self, anomaly_errors: List[AnomalyError], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of anomaly detection results.
        
        Args:
            anomaly_errors: List of AnomalyError objects from an anomaly detector
            original_df: Original dataframe that was analyzed
             
        Returns:
            List of reports with human-readable anomaly messages
        """
        report = []
        for error in anomaly_errors:
            display_message = self._format_error_message(error)

            report.append({
                "row_index": error.row_index,
                "error_data": error.anomaly_data,
                "display_message": display_message,
                "column_name": error.column_name,
                "probability": error.probability,
                "detection_type": "pattern_based",  # Indicate this is pattern-based detection
                "anomaly_code": error.anomaly_type
            })
        return report
