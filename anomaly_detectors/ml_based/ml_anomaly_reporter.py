"""
ML-based anomaly reporter implementation following the standard pattern.
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd

from anomaly_detectors.anomaly_error import AnomalyError
from anomaly_detectors.reporter_interface import AnomalyReporterInterface


class MLAnomalyReporter(AnomalyReporterInterface):
    """
    Implements the AnomalyReporterInterface to translate ML-based anomaly detection
    results into human-readable messages.
    """

    def __init__(self, include_technical_details: bool = False):
        """
        Initialize the ML anomaly reporter.

        Args:
            include_technical_details (bool): Whether to include technical details
                                            like similarity scores in the reports.
        """
        self.include_technical_details = include_technical_details
        self.error_messages = self._load_error_messages()

    def _load_error_messages(self) -> Dict[str, str]:
        """Load error message templates for ML-based anomalies."""
        try:
            # Try to load from JSON file
            with open(os.path.join(os.path.dirname(__file__), 'ml_explanation_templates.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Use default templates if file doesn't exist
            return {
                "SIMILARITY_ANOMALY": "Value '{value}' appears unusual based on ML similarity analysis (probability: {probability:.3f})",
                "OUTLIER_DETECTED": "Value '{value}' detected as outlier by ML model (probability: {probability:.3f})",
                "PATTERN_DEVIATION": "Value '{value}' deviates from learned patterns (probability: {probability:.3f})",
                "UNKNOWN_PATTERN": "Value '{value}' doesn't match any known patterns (probability: {probability:.3f})"
            }

    def generate_report(self, anomaly_errors: List[AnomalyError], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate human-readable reports from ML-based anomaly errors.

        Args:
            anomaly_errors (List[AnomalyError]): List of anomaly errors to report
            data (pd.DataFrame): The original data being analyzed

        Returns:
            List[Dict[str, Any]]: List of report dictionaries with human-readable messages
        """
        reports = []

        for error in anomaly_errors:
            report = self._create_report_for_error(error, data)
            reports.append(report)

        return reports

    def _create_report_for_error(self, error: AnomalyError, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a report dictionary for a single anomaly error.

        Args:
            error (AnomalyError): The anomaly error to report
            data (pd.DataFrame): The original data being analyzed

        Returns:
            Dict[str, Any]: Report dictionary with display message and metadata
        """
        # Get the error message template
        template = self.error_messages.get(error.error_code,
                                         "Value '{value}' flagged as anomaly (probability: {probability:.3f})")

        # Format the message with error details
        display_message = template.format(
            value=error.value,
            probability=error.probability,
            **error.metadata
        )

        # Create the base report
        report = {
            'row_index': error.row_index,
            'column_name': error.column_name,
            'value': error.value,
            'error_code': error.error_code,
            'probability': error.probability,
            'display_message': display_message,
            'is_ml_based': True
        }

        # Add technical details if requested
        if self.include_technical_details:
            technical_details = {
                'similarity_score': error.metadata.get('similarity_score', 'N/A'),
                'nearest_neighbors': error.metadata.get('nearest_neighbors', []),
                'model_type': error.metadata.get('model_type', 'Unknown'),
                'threshold': error.metadata.get('threshold', 'N/A')
            }
            report['technical_details'] = technical_details

        return report

    def get_summary_statistics(self, anomaly_errors: List[AnomalyError]) -> Dict[str, Any]:
        """
        Generate summary statistics for ML-based anomaly errors.

        Args:
            anomaly_errors (List[AnomalyError]): List of anomaly errors

        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not anomaly_errors:
            return {
                'total_anomalies': 0,
                'avg_probability': 0.0,
                'error_code_distribution': {},
                'columns_affected': []
            }

        # Calculate statistics
        total_anomalies = len(anomaly_errors)
        avg_probability = sum(error.probability for error in anomaly_errors) / total_anomalies

        # Error code distribution
        error_code_counts = {}
        for error in anomaly_errors:
            error_code_counts[error.error_code] = error_code_counts.get(error.error_code, 0) + 1

        # Columns affected
        columns_affected = list(set(error.column_name for error in anomaly_errors))

        return {
            'total_anomalies': total_anomalies,
            'avg_probability': avg_probability,
            'error_code_distribution': error_code_counts,
            'columns_affected': columns_affected
        }
