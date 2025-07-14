import pandas as pd
import json
import os
from typing import List, Dict, Any
from anomaly_detectors.reporter_interface import AnomalyReporterInterface
from anomaly_detectors.anomaly_error import AnomalyError

class AnomalyReporter(AnomalyReporterInterface):
    """
    Implements the ReporterInterface to translate structured anomaly detection
    results into human-readable messages for any anomaly detector.
    """

    def __init__(self, detector_name):
        """
        Initialize the reporter and load error messages from the JSON file.
        
        Args:
            detector_name (str): The name of the anomaly detector to load error messages for.
                               This is mandatory and must be provided.
        """
        # Load error messages from the JSON file
        if detector_name is None or detector_name.strip() == "":
            print("Error: detector_name is required for AnomalyReporter initialization")
            import sys
            sys.exit(1)
            
        error_messages_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "anomaly_detectors", detector_name, "error_messages.json"
        )
        try:
            with open(error_messages_path, 'r') as f:
                self.ERROR_MESSAGES = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find error messages file at {error_messages_path}")
            print(f"Please ensure that anomaly detector '{detector_name}' exists and has an error_messages.json file.")
            import sys
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The error messages file at {error_messages_path} contains invalid JSON.")
            import sys
            sys.exit(1)

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
            if error_code in self.ERROR_MESSAGES:
                message_template = self.ERROR_MESSAGES[error_code]
            elif "DEFAULT" in self.ERROR_MESSAGES:
                message_template = self.ERROR_MESSAGES["DEFAULT"]
            else:
                message_template = "Anomaly detected with data: {error_data}"
            
            # Format the message with specific details from the error
            details = error.details.copy() if error.details else {}  # Create a copy to avoid modifying the original
            details['error_data'] = error.anomaly_data  # Add original data for context
            details['confidence'] = error.anomaly_score  # For backward compatibility, provide confidence as alias to anomaly_score
            details['anomaly_score'] = error.anomaly_score  # Add anomaly_score for potential use in message
            
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
                "confidence": error.anomaly_score,    # For backward compatibility, add as confidence
                "anomaly_score": error.anomaly_score,  # Add anomaly_score to the report
                "anomaly": True                    # Flag to indicate this is an anomaly, not a validation error
            })
        return report
