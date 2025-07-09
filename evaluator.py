import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List, Dict, Any, Union, Optional, Tuple

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.reporter_interface import ReporterInterface
from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface


class Evaluator:
    """
    A class that orchestrates the validation and anomaly detection process on data.
    It coordinates between validators, anomaly detectors, and reporters to produce a comprehensive
    analysis of data quality issues.
    """
    
    def __init__(self, 
                 validator: Optional[ValidatorInterface] = None,
                 validator_reporter: Optional[ReporterInterface] = None,
                 anomaly_detector: Optional[AnomalyDetectorInterface] = None,
                 anomaly_reporter: Optional[ReporterInterface] = None):
        """
        Initialize the Evaluator with validators and anomaly detectors.
        
        Args:
            validator: Optional validator to check for data validation errors
            validator_reporter: Optional reporter to format validation errors
            anomaly_detector: Optional anomaly detector to check for anomalies
            anomaly_reporter: Optional reporter to format anomaly detection results
        """
        self.validator = validator
        self.validator_reporter = validator_reporter
        self.anomaly_detector = anomaly_detector
        self.anomaly_reporter = anomaly_reporter
        
        # Log which components are active
        components = []
        if validator:
            components.append("Validator")
        if anomaly_detector:
            components.append("Anomaly Detector")
        
        if components:
            print(f"Evaluator initialized with: {', '.join(components)}")
        
    def evaluate_column(self, 
                       df: pd.DataFrame, 
                       column_name: str,
                       run_validation: bool = True,
                       run_anomaly_detection: bool = True) -> Dict[str, Any]:
        """
        Evaluates a column in the dataframe using the configured validator and anomaly detector.
        
        Args:
            df: The dataframe to evaluate
            column_name: The name of the column to evaluate
            run_validation: Whether to run validation
            run_anomaly_detection: Whether to run anomaly detection
            
        Returns:
            A dictionary containing validation results and anomaly detection results
        """
        results = {
            "column_name": column_name,
            "row_count": len(df),
            "validation_results": [],
            "anomaly_results": [],
            "total_validation_errors": 0,
            "total_anomalies": 0
        }
        
        # Run validation if requested and validator is available
        if run_validation and self.validator and self.validator_reporter:
            validation_errors = self.validator.bulk_validate(df, column_name)
            validation_report = self.validator_reporter.generate_report(validation_errors, df)
            results["validation_results"] = validation_report
            results["total_validation_errors"] = len(validation_errors)
        
        # Run anomaly detection if requested and detector is available
        if run_anomaly_detection and self.anomaly_detector and self.anomaly_reporter:
            anomalies = self.anomaly_detector.bulk_detect(df, column_name)
            anomaly_report = self.anomaly_reporter.generate_report(anomalies, df)
            results["anomaly_results"] = anomaly_report
            results["total_anomalies"] = len(anomalies)
            
        return results
        
    def evaluate_sample(self, 
                        sample_df: pd.DataFrame, 
                        column_name: str,
                        injected_errors: List[Dict[str, Any]] = None,
                        run_validation: bool = True,
                        run_anomaly_detection: bool = True) -> Dict[str, Any]:
        """
        Evaluates a sample including validation, anomaly detection, and measuring performance.
        
        Args:
            sample_df: The sample dataframe to evaluate
            column_name: The name of the column to evaluate
            injected_errors: Optional list of injected errors for measuring validator performance
            run_validation: Whether to run validation
            run_anomaly_detection: Whether to run anomaly detection
            
        Returns:
            A dictionary containing the evaluation results including metrics
        """
        # Evaluate the sample
        results = self.evaluate_column(
            sample_df, 
            column_name, 
            run_validation=run_validation,
            run_anomaly_detection=run_anomaly_detection
        )
        
        # Add evaluation time
        results["sample_id"] = getattr(sample_df, "name", None)
        
        # If injected errors are provided, calculate performance metrics
        if injected_errors and run_validation and self.validator:
            metrics = self._calculate_metrics(
                sample_df, 
                column_name, 
                results["validation_results"], 
                injected_errors
            )
            results.update(metrics)
            
        return results
    
    def _calculate_metrics(self, 
                          df: pd.DataFrame, 
                          column_name: str,
                          validation_results: List[Dict[str, Any]],
                          injected_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates performance metrics by comparing detected errors with injected errors.
        
        Args:
            df: The dataframe that was validated
            column_name: The name of the column that was validated
            validation_results: The results from validation
            injected_errors: The list of injected errors
            
        Returns:
            A dictionary containing performance metrics
        """
        # Extract row indices from validation results
        detected_row_indices = set(result["row_index"] for result in validation_results)
        
        # Extract row indices from injected errors
        error_row_indices = set(error["row_index"] for error in injected_errors)
        
        # Calculate true positives, false positives, and false negatives
        true_positives = detected_row_indices.intersection(error_row_indices)
        false_positives = detected_row_indices - error_row_indices
        false_negatives = error_row_indices - detected_row_indices
        
        # Calculate metrics
        precision = len(true_positives) / len(detected_row_indices) if detected_row_indices else 1.0
        recall = len(true_positives) / len(error_row_indices) if error_row_indices else 1.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        # Detailed information about errors
        true_positive_details = [
            {
                "row_index": result["row_index"],
                "error_data": result["error_data"],
                "display_message": result["display_message"]
            }
            for result in validation_results
            if result["row_index"] in true_positives
        ]
        
        false_positive_details = [
            {
                "row_index": result["row_index"],
                "error_data": result["error_data"],
                "display_message": result["display_message"]
            }
            for result in validation_results
            if result["row_index"] in false_positives
        ]
        
        false_negative_details = [
            {
                "row_index": error["row_index"],
                "error_data": df.iloc[error["row_index"]][column_name],
                "injected_error": error
            }
            for error in injected_errors
            if error["row_index"] in false_negatives
        ]
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positive_details,
            "false_positives": false_positive_details,
            "false_negatives": false_negative_details,
            "true_positive_count": len(true_positives),
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "validation_performed": True  # Flag to indicate validation was performed
        }
