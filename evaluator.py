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
from anomaly_detectors.reporter_interface import AnomalyReporterInterface
from unified_detection_interface import (
    UnifiedDetectionResult, 
    CombinedDetector, 
    UnifiedReporter,
    DetectionType
)
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
from anomaly_detectors.ml_based.ml_anomaly_reporter import MLAnomalyReporter


class Evaluator:
    """
    A class that orchestrates the validation and anomaly detection process on data.
    It coordinates between validators, anomaly detectors, ML detectors, and reporters 
    to produce a comprehensive analysis of data quality issues using a unified interface.
    """
    
    def __init__(self, 
                 validator: Optional[ValidatorInterface] = None,
                 validator_reporter: Optional[ReporterInterface] = None,
                 anomaly_detector: Optional[AnomalyDetectorInterface] = None,
                 anomaly_reporter: Optional[AnomalyReporterInterface] = None,
                 ml_detector: Optional[MLAnomalyDetector] = None,
                 ml_reporter: Optional[MLAnomalyReporter] = None):
        """
        Initialize the Evaluator with validators, anomaly detectors, and ML detectors.
        
        Args:
            validator: Optional validator to check for data validation errors
            validator_reporter: Optional reporter to format validation errors
            anomaly_detector: Optional anomaly detector to check for anomalies
            anomaly_reporter: Optional reporter to format anomaly detection results
            ml_detector: Optional ML-based anomaly detector
            ml_reporter: Optional reporter to format ML-based anomaly detection results
        """
        self.validator = validator
        self.validator_reporter = validator_reporter
        self.anomaly_detector = anomaly_detector
        self.anomaly_reporter = anomaly_reporter
        self.ml_detector = ml_detector
        self.ml_reporter = ml_reporter
        
        # Create unified components
        self.combined_detector = CombinedDetector(
            validator=validator,
            anomaly_detector=anomaly_detector,
            ml_detector=ml_detector
        )
        self.unified_reporter = UnifiedReporter(include_technical_details=True)
        
        # Log which components are active
        components = []
        if validator:
            components.append("Validator" + (" + Reporter" if validator_reporter else ""))
        if anomaly_detector:
            components.append("Anomaly Detector" + (" + Reporter" if anomaly_reporter else ""))
        if ml_detector:
            components.append("ML Detector" + (" + Reporter" if ml_reporter else ""))
        
        if components:
            print(f"Evaluator initialized with: {', '.join(components)}")
        else:
            print("Warning: Evaluator initialized with no detection components")
    
    def evaluate_unified(self, 
                        df: pd.DataFrame, 
                        column_name: str,
                        detection_types: Optional[List[DetectionType]] = None,
                        threshold: float = 0.7) -> Dict[str, Any]:
        """
        Evaluates a column using the unified detection interface that combines all approaches.
        
        Args:
            df: The dataframe to evaluate
            column_name: The name of the column to evaluate
            detection_types: Optional list of detection types to run (defaults to all available)
            threshold: Threshold for anomaly detection scoring
            
        Returns:
            A dictionary containing unified detection results
        """
        # Run unified detection
        enable_validation = detection_types is None or DetectionType.VALIDATION in detection_types
        enable_anomaly = detection_types is None or DetectionType.ANOMALY in detection_types  
        enable_ml = detection_types is None or DetectionType.ML_ANOMALY in detection_types
        
        unified_results = self.combined_detector.detect_issues(
            df, 
            column_name,
            enable_validation=enable_validation,
            enable_anomaly_detection=enable_anomaly,
            enable_ml_detection=enable_ml,
            anomaly_threshold=threshold,
            ml_threshold=threshold
        )
        
        # Generate unified reports
        unified_reports = self.unified_reporter.generate_report(
            unified_results, 
            df
        )
        
        # Organize results by detection type
        results = {
            "column_name": column_name,
            "row_count": len(df),
            "unified_results": unified_reports,
            "total_issues": len(unified_reports),
            "issues_by_type": {}
        }
        
        # Count issues by detection type
        for report in unified_reports:
            detection_type = report.get("detection_type", "unknown")
            if detection_type not in results["issues_by_type"]:
                results["issues_by_type"][detection_type] = 0
            results["issues_by_type"][detection_type] += 1
        
        return results        
    def evaluate_column_unified(self, 
                               df: pd.DataFrame, 
                               column_name: str,
                               enable_validation: bool = True,
                               enable_anomaly_detection: bool = True,
                               enable_ml_detection: bool = True,
                               validation_threshold: float = 0.0,
                               anomaly_threshold: float = 0.7,
                               ml_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Evaluates a column using the unified detection interface.
        
        Args:
            df: The dataframe to evaluate
            column_name: The name of the column to evaluate
            enable_validation: Whether to run validation
            enable_anomaly_detection: Whether to run pattern-based anomaly detection
            enable_ml_detection: Whether to run ML-based anomaly detection
            validation_threshold: Minimum probability for validation results
            anomaly_threshold: Minimum probability for anomaly detection results
            ml_threshold: Minimum probability for ML detection results
            
        Returns:
            A dictionary containing unified detection results and analysis
        """
        # Run unified detection
        detection_results = self.combined_detector.detect_issues(
            df=df,
            column_name=column_name,
            enable_validation=enable_validation,
            enable_anomaly_detection=enable_anomaly_detection,
            enable_ml_detection=enable_ml_detection,
            validation_threshold=validation_threshold,
            anomaly_threshold=anomaly_threshold,
            ml_threshold=ml_threshold
        )
        
        # Generate unified report
        unified_report = self.unified_reporter.generate_report(detection_results, df)
        
        # Generate summary
        summary = self.unified_reporter.generate_summary(detection_results)
        
        # Organize results by detection type
        results_by_type = {
            'validation': [],
            'anomaly': [],
            'ml_anomaly': []
        }
        
        for result in detection_results:
            if result.detection_type == DetectionType.VALIDATION:
                results_by_type['validation'].append(result.to_dict())
            elif result.detection_type == DetectionType.ANOMALY:
                results_by_type['anomaly'].append(result.to_dict())
            elif result.detection_type == DetectionType.ML_ANOMALY:
                results_by_type['ml_anomaly'].append(result.to_dict())
        
        return {
            "column_name": column_name,
            "row_count": len(df),
            "unified_report": unified_report,
            "summary": summary,
            "results_by_type": results_by_type,
            "detection_results": [r.to_dict() for r in detection_results],
            "total_issues": len(detection_results),
            "thresholds_used": {
                "validation": validation_threshold,
                "anomaly": anomaly_threshold,
                "ml": ml_threshold
            }
        }
    
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
    
    def _evaluate_ml_individual(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Evaluate ML detection individually to get separate results.
        
        Args:
            df: The dataframe to evaluate
            column_name: The name of the column to evaluate
            
        Returns:
            A dictionary containing ML detection results
        """
        ml_results = {
            "ml_results": [],
            "total_ml_issues": 0
        }
        
        if self.ml_detector and hasattr(self.ml_detector, 'bulk_detect'):
            try:
                ml_anomalies = self.ml_detector.bulk_detect(df, column_name)
                
                # Use ML reporter if available, otherwise format manually
                if self.ml_reporter:
                    ml_reports = self.ml_reporter.generate_report(ml_anomalies, df)
                else:
                    # Fallback to manual formatting for backward compatibility
                    ml_reports = []
                    for ml_result in ml_anomalies:
                        ml_report = {
                            "row_index": ml_result.row_index,
                            "column_name": ml_result.column_name,
                            "value": ml_result.value,
                            "probability": ml_result.probability,
                            "error_code": ml_result.error_code,
                            "display_message": f"ML anomaly detected with probability {ml_result.probability:.2f}",
                            "ml_features": ml_result.metadata
                        }
                        ml_reports.append(ml_report)
                
                ml_results["ml_results"] = ml_reports
                ml_results["total_ml_issues"] = len(ml_reports)
                
            except Exception as e:
                print(f"Warning: Individual ML detection failed: {e}")
                
        return ml_results
        
    def evaluate_sample(self, 
                        sample_df: pd.DataFrame, 
                        column_name: str,
                        injected_errors: List[Dict[str, Any]] = None,
                        run_validation: bool = True,
                        run_anomaly_detection: bool = True,
                        use_unified_approach: bool = False) -> Dict[str, Any]:
        """
        Evaluates a sample including validation, anomaly detection, ML detection, and measuring performance.
        
        Args:
            sample_df: The sample dataframe to evaluate
            column_name: The name of the column to evaluate
            injected_errors: Optional list of injected errors for measuring validator performance
            run_validation: Whether to run validation
            run_anomaly_detection: Whether to run anomaly detection
            use_unified_approach: Whether to also use the unified detection approach
            
        Returns:
            A dictionary containing comprehensive evaluation results including individual and combined approaches
        """
        
        # If using unified approach, run unified detection to avoid duplication
        if use_unified_approach:
            # Run unified approach only
            detection_types = []
            if run_validation and self.validator:
                detection_types.append(DetectionType.VALIDATION)
            if run_anomaly_detection and self.anomaly_detector:
                detection_types.append(DetectionType.ANOMALY)
            if self.ml_detector:
                detection_types.append(DetectionType.ML_ANOMALY)
            
            unified_results = self.evaluate_unified(
                sample_df, 
                column_name, 
                detection_types=detection_types
            )
            
            # Convert unified results to individual format for compatibility
            comprehensive_results = {
                "column_name": column_name,
                "row_count": len(sample_df),
                "sample_id": getattr(sample_df, "name", None),
                
                # Convert unified results to individual format
                "validation_results": [r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "validation"],
                "anomaly_results": [r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "anomaly"],
                "ml_results": [r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "ml_anomaly"],
                
                # Count individual approach results
                "total_validation_errors": len([r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "validation"]),
                "total_anomalies": len([r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "anomaly"]),
                "total_ml_issues": len([r for r in unified_results.get("unified_results", []) if r.get("detection_type") == "ml_anomaly"]),
                
                # Approach availability flags
                "validation_available": run_validation and self.validator is not None,
                "anomaly_detection_available": run_anomaly_detection and self.anomaly_detector is not None,
                "ml_detection_available": self.ml_detector is not None,
                
                # Unified approach results
                "unified_approach_used": True,
                "unified_results": unified_results.get("unified_results", []),
                "unified_total_issues": unified_results.get("total_issues", 0),
                "unified_issues_by_type": unified_results.get("issues_by_type", {}),
            }
        else:
            # Run individual approaches separately (original behavior)
            individual_results = self.evaluate_column(
                sample_df, 
                column_name, 
                run_validation=run_validation,
                run_anomaly_detection=run_anomaly_detection
            )
            
            # Run ML detection individually if available
            ml_individual_results = {}
            if self.ml_detector:
                ml_individual_results = self._evaluate_ml_individual(sample_df, column_name)
            
            # Combine all individual results
            comprehensive_results = {
                "column_name": column_name,
                "row_count": len(sample_df),
                "sample_id": getattr(sample_df, "name", None),
                
                # Individual approach results
                "validation_results": individual_results.get("validation_results", []),
                "anomaly_results": individual_results.get("anomaly_results", []),
                "ml_results": ml_individual_results.get("ml_results", []),
                
                # Individual approach counts
                "total_validation_errors": individual_results.get("total_validation_errors", 0),
                "total_anomalies": individual_results.get("total_anomalies", 0),
                "total_ml_issues": ml_individual_results.get("total_ml_issues", 0),
                
                # Approach availability flags
                "validation_available": run_validation and self.validator is not None,
                "anomaly_detection_available": run_anomaly_detection and self.anomaly_detector is not None,
                "ml_detection_available": self.ml_detector is not None,
                
                # Unified approach not used
                "unified_approach_used": False
            }
        
        # Calculate performance metrics if injected errors are provided
        if injected_errors and run_validation and self.validator:
            validation_results_for_metrics = comprehensive_results["validation_results"]
            metrics = self._calculate_metrics(
                sample_df, 
                column_name, 
                validation_results_for_metrics, 
                injected_errors
            )
            comprehensive_results.update(metrics)
            
        return comprehensive_results
    
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
        detected_row_indices = set()
        for result in validation_results:
            if "row_index" in result:
                detected_row_indices.add(result["row_index"])
        
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
        true_positive_details = []
        false_positive_details = []
        
        for result in validation_results:
            if result.get("row_index") in true_positives:
                true_positive_details.append({
                    "row_index": result.get("row_index"),
                    "error_data": result.get("error_data") or result.get("value"),
                    "display_message": result.get("display_message", "")
                })
            elif result.get("row_index") in false_positives:
                false_positive_details.append({
                    "row_index": result.get("row_index"),
                    "error_data": result.get("error_data") or result.get("value"),
                    "display_message": result.get("display_message", "")
                })
        
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
