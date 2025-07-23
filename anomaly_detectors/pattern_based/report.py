import pandas as pd
import json
import os
import sys
from typing import List, Dict, Any
from anomaly_detectors.reporter_interface import AnomalyReporterInterface
from anomaly_detectors.anomaly_error import AnomalyError

# Add the parent directory to the path to import our exception module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from exceptions import ConfigurationError, FileOperationError

class AnomalyReporter(AnomalyReporterInterface):
    """
    Implements the AnomalyReporterInterface to translate structured anomaly
    detection results into human-readable messages for any anomaly detector.
    """

    def __init__(self, detector_name):
        """
        Initialize the reporter and load error messages from the JSON file.
        
        Args:
            detector_name (str): The name of the detector to load error messages for.
        """
        self.detector_name = detector_name
        try:
            self.error_messages = self._load_error_messages(detector_name)
        except FileNotFoundError:
            raise FileOperationError(
                f"Error messages file not found for detector '{detector_name}'",
                details={'detector_name': detector_name}
            )
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in error messages file for detector '{detector_name}'",
                details={'detector_name': detector_name, 'json_error': str(e)}
            ) from e

    def _load_error_messages(self, detector_name):
        """Load error messages from the JSON file."""
        error_messages_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            detector_name, "error_messages.json"
        )
        
        try:
            with open(error_messages_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileOperationError(
                f"Error messages file not found for detector '{detector_name}'",
                details={'detector_name': detector_name}
            )
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in error messages file for detector '{detector_name}'",
                details={'detector_name': detector_name, 'json_error': str(e)}
            ) from e

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
            error_code = error.anomaly_type
            
            # First try to get the template for the specific error code
            # If not found, try to use DEFAULT, and if that's not available, use a generic message
            if error_code in self.error_messages:
                message_template = self.error_messages[error_code]
            elif "DEFAULT" in self.error_messages:
                message_template = self.error_messages["DEFAULT"]
            else:
                message_template = "Anomaly detected with data: {error_data}"
            
            # Format the message with specific details from the error
            details = error.details.copy() if error.details else {}  # Create a copy to avoid modifying the original
            details['error_data'] = error.anomaly_data  # Add original data for context
            details['probability'] = error.probability  # Add probability for potential use in message
            
            try:
                display_message = message_template.format(**details)
            except KeyError as e:
                # Handle missing format keys gracefully
                display_message = f"Error formatting message: {message_template} (missing {e})"
            except Exception as e:
                display_message = f"Error formatting message: {str(e)}"

            report.append({
                "row_index": error.row_index,
                "error_data": error.anomaly_data,
                "display_message": display_message,
                "column_name": error.column_name,  # Add column name to the report
                "probability": error.probability,    # Add probability to the report
                "anomaly": True                    # Flag to indicate this is an anomaly, not a validation error
            })
        return report
