import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_detection_interface import CombinedDetector, DetectionConfig, DetectionType, UnifiedReporter

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
from anomaly_detectors.ml_based.ml_anomaly_reporter import MLAnomalyReporter
from anomaly_detectors.reporter_interface import AnomalyReporterInterface
from common.field_mapper import FieldMapper
from validators.reporter_interface import ReporterInterface
from validators.validator_interface import ValidatorInterface


class Evaluator:
    """
    A class that orchestrates the validation and anomaly detection process on data.
    It coordinates between validators, anomaly detectors, ML detectors, and reporters
    to produce a comprehensive analysis of data quality issues using a unified interface.
    """

    def __init__(self,
                 high_confidence_threshold: float,
                 batch_size: Optional[int],
                 max_workers: int,
                 validator: Optional[ValidatorInterface] = None,
                 validator_reporter: Optional[ReporterInterface] = None,
                 anomaly_detector: Optional[AnomalyDetectorInterface] = None,
                 anomaly_reporter: Optional[AnomalyReporterInterface] = None,
                 ml_detector: Optional[MLAnomalyDetector] = None,
                 ml_reporter: Optional[MLAnomalyReporter] = None,
                 field_mapper: Optional[FieldMapper] = None):
        """
        Initialize the Evaluator with validators, anomaly detectors, and ML detectors.

        Args:
            validator: Optional validator to check for data validation errors
            validator_reporter: Optional reporter to format validation errors
            anomaly_detector: Optional anomaly detector to check for anomalies
            anomaly_reporter: Optional reporter to format anomaly detection results
            ml_detector: Optional ML-based anomaly detector
            ml_reporter: Optional reporter to format ML-based anomaly detection results
            field_mapper: Optional field mapper service (uses default if not provided)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers

        self.validator = validator
        self.validator_reporter = validator_reporter
        self.anomaly_detector = anomaly_detector
        self.anomaly_reporter = anomaly_reporter
        self.ml_detector = ml_detector
        self.ml_reporter = ml_reporter

        # Use provided field mapper or create default one
        self.field_mapper = field_mapper
        if self.field_mapper is None:
            raise ValueError("field_mapper must be provided")

        # Create combined detector for unified approach
        self.combined_detector = CombinedDetector(
            field_mapper=self.field_mapper,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            validator=validator,
            anomaly_detector=anomaly_detector,
            ml_detector=ml_detector
        )
        # Create unified reporter for combined detection approach
        self.unified_reporter = UnifiedReporter(high_confidence_threshold, include_technical_details=True)

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
                        field_name: str,
                        detection_types: Optional[List[DetectionType]] = None,
                        validation_threshold: float = None,
                        anomaly_threshold: float = None,
                        ml_threshold: float = None) -> Dict[str, Any]:
        """
        Evaluates a field using the unified detection interface that combines all approaches.

        Args:
            df: The dataframe to evaluate
            field_name: The name of the field to evaluate
            detection_types: Optional list of detection types to run (defaults to all available)
            validation_threshold: Threshold for validation results (required)
            anomaly_threshold: Threshold for anomaly detection (required)
            ml_threshold: Threshold for ML detection (required)

        Returns:
            A dictionary containing unified detection results
        """
        # Require explicit threshold values - no hardcoded fallbacks
        config = DetectionConfig(
            validation_threshold=validation_threshold,
            anomaly_threshold=anomaly_threshold,
            ml_threshold=ml_threshold,
            enable_validation=detection_types is None or DetectionType.VALIDATION in detection_types,
            enable_anomaly_detection=detection_types is None or DetectionType.ANOMALY in detection_types,
            enable_ml_detection=detection_types is None or DetectionType.ML_ANOMALY in detection_types
        )

        # Run unified detection
        unified_results = self.combined_detector.detect_issues(df, field_name, config)

        # Generate unified reports
        unified_reports = self.unified_reporter.generate_report(unified_results, df)

        # Organize results by detection type
        results = {
            "field_name": field_name,
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

    def evaluate_field(self,
                       df: pd.DataFrame,
                       field_name: str,
                       run_validation: bool,
                       run_anomaly_detection: bool) -> Dict[str, Any]:
        """
        Evaluates a field in the dataframe using the configured validator and anomaly detector.

        Args:
            df: The dataframe to evaluate
            field_name: The name of the field to evaluate
            run_validation: Whether to run validation
            run_anomaly_detection: Whether to run anomaly detection

        Returns:
            A dictionary containing validation results and anomaly detection results
        """
        # Get the column name using centralized field mapping
        column_name = self.field_mapper.validate_column_exists(df, field_name)

        results = {
            "field_name": field_name,
            "column_name": column_name,  # Keep for debugging/reference
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
            anomalies = self.anomaly_detector.bulk_detect(df, column_name, self.batch_size, self.max_workers)
            anomaly_report = self.anomaly_reporter.generate_report(anomalies, df)
            results["anomaly_results"] = anomaly_report
            results["total_anomalies"] = len(anomalies)

        return results

    def _evaluate_ml_individual(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Evaluate ML detection individually to get separate results.

        Args:
            df: The dataframe to evaluate
            field_name: The name of the field to evaluate

        Returns:
            A dictionary containing ML detection results
        """
        # Get the column name using centralized field mapping
        column_name = self.field_mapper.validate_column_exists(df, field_name)

        ml_results = {
            "ml_results": [],
            "total_ml_issues": 0
        }

        if self.ml_detector and hasattr(self.ml_detector, 'bulk_detect'):
            try:
                ml_anomalies = self.ml_detector.bulk_detect(df, column_name, self.batch_size, self.max_workers)

                # Use ML reporter if available, otherwise format manually
                if self.ml_reporter:
                    ml_reports = self.ml_reporter.generate_report(ml_anomalies, df)
                else:
                    # Fallback to manual formatting for backward compatibility
                    ml_reports = []
                    for ml_result in ml_anomalies:
                        ml_report = {
                            "row_index": ml_result.row_index,
                            "field_name": ml_result.field_name,
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
                        field_name: str,
                        injected_errors: List[Dict[str, Any]],
                        run_validation: bool,
                        run_anomaly_detection: bool,
                        use_unified_approach: bool,
                        validation_threshold: float = None,
                        anomaly_threshold: float = None,
                        ml_threshold: float = None) -> Dict[str, Any]:
        """
        Evaluates a sample including validation, anomaly detection, ML detection, and measuring performance.

        Args:
            sample_df: The sample dataframe to evaluate
            field_name: The name of the field to evaluate
            injected_errors: List of injected errors for measuring validator performance
            run_validation: Whether to run validation
            run_anomaly_detection: Whether to run anomaly detection
            use_unified_approach: Whether to also use the unified detection approach
            validation_threshold: Threshold for validation results
            anomaly_threshold: Threshold for anomaly detection
            ml_threshold: Threshold for ML detection

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
                field_name,
                detection_types=detection_types,
                validation_threshold=validation_threshold,
                anomaly_threshold=anomaly_threshold,
                ml_threshold=ml_threshold
            )

            # Convert unified results to individual format for compatibility
            comprehensive_results = {
                "field_name": field_name,
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
            individual_results = self.evaluate_field(
                sample_df,
                field_name,
                run_validation=run_validation,
                run_anomaly_detection=run_anomaly_detection
            )

            # Run ML detection individually if available
            ml_individual_results = {}
            if self.ml_detector:
                ml_individual_results = self._evaluate_ml_individual(sample_df, field_name)

            # Combine all individual results
            comprehensive_results = {
                "field_name": field_name,
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
                field_name,
                validation_results_for_metrics,
                injected_errors
            )
            comprehensive_results.update(metrics)

        return comprehensive_results

    def _calculate_metrics(self,
                          df: pd.DataFrame,
                          field_name: str,
                          validation_results: List[Dict[str, Any]],
                          injected_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics for validation results against injected errors.

        Args:
            df: The dataframe that was validated
            field_name: The field name that was validated
            validation_results: Results from validation
            injected_errors: List of injected errors for comparison
        """
        # Get the column name using centralized field mapping
        column_name = self.field_mapper.validate_column_exists(df, field_name)

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
