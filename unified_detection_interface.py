"""
Unified Detection Interface for Data Quality Monitoring

This module provides a unified interface that combines validation, anomaly detection,
and ML-based detection approaches into a single, consistent framework.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json

from validators.validation_error import ValidationError
from anomaly_detectors.anomaly_error import AnomalyError
from anomaly_detectors.reporter_interface import MLAnomalyResult


class DetectionType(str, Enum):
    """Enumeration for different types of detection approaches."""
    VALIDATION = "validation"      # Rule-based validation (high confidence)
    ANOMALY = "anomaly"           # Pattern-based anomaly detection (medium confidence)
    ML_ANOMALY = "ml_anomaly"     # ML-based anomaly detection (variable confidence)


class UnifiedDetectionResult:
    """
    A unified result class that can represent any type of detection result.
    This provides a consistent interface across all detection approaches.
    """
    
    def __init__(self,
                 row_index: int,
                 column_name: str,
                 value: Any,
                 detection_type: DetectionType,
                 probability: float,
                 error_code: str,
                 message: str,
                 details: Dict[str, Any] = None,
                 ml_features: Dict[str, Any] = None):
        """
        Initialize a unified detection result.
        
        Args:
            row_index: The index of the row where the issue was detected
            column_name: The name of the column where the issue was detected
            value: The original data value that was flagged
            detection_type: The type of detection that found this issue
            probability: A value between 0 and 1 indicating probability that this is an issue
            error_code: A string code identifying the type of error/anomaly
            message: A human-readable message explaining the issue
            details: Optional dictionary containing additional details
            ml_features: Optional ML-specific features (embeddings, probabilities, etc.)
        """
        self.row_index = row_index
        self.column_name = column_name
        self.value = value
        self.detection_type = detection_type
        self.probability = probability
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.ml_features = ml_features or {}
    
    @classmethod
    def from_validation_error(cls, validation_error: ValidationError) -> 'UnifiedDetectionResult':
        """Create a UnifiedDetectionResult from a ValidationError."""
        return cls(
            row_index=validation_error.row_index,
            column_name=validation_error.column_name,
            value=validation_error.error_data,
            detection_type=DetectionType.VALIDATION,
            probability=validation_error.probability,
            error_code=str(validation_error.error_type),
            message=f"Validation error: {validation_error.error_type}",
            details=validation_error.details
        )
    
    @classmethod
    def from_anomaly_error(cls, anomaly_error: AnomalyError) -> 'UnifiedDetectionResult':
        """Create a UnifiedDetectionResult from an AnomalyError."""
        return cls(
            row_index=anomaly_error.row_index,
            column_name=anomaly_error.column_name,
            value=anomaly_error.anomaly_data,
            detection_type=DetectionType.ANOMALY,
            probability=anomaly_error.probability,
            error_code=str(anomaly_error.anomaly_type),
            message=f"Anomaly detected: {anomaly_error.anomaly_type}",
            details=anomaly_error.details
        )
    
    @classmethod
    def from_ml_anomaly_result(cls, ml_result: MLAnomalyResult) -> 'UnifiedDetectionResult':
        """Create a UnifiedDetectionResult from an MLAnomalyResult."""
        ml_features = {
            'anomaly_probability': ml_result.anomaly_score,  # Keep old name for backward compatibility
            'feature_contributions': ml_result.feature_contributions,
            'nearest_neighbors': ml_result.nearest_neighbors,
            'cluster_info': ml_result.cluster_info,
            'probability_info': ml_result.probability_info
        }
        
        message = ml_result.explanation or f"ML anomaly detected with probability {ml_result.anomaly_score:.2f}"
        
        return cls(
            row_index=ml_result.row_index,
            column_name=ml_result.column_name,
            value=ml_result.value,
            detection_type=DetectionType.ML_ANOMALY,
            probability=ml_result.anomaly_score,
            error_code="ML_ANOMALY",
            message=message,
            details={'explanation': ml_result.explanation},
            ml_features=ml_features
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary format."""
        return {
            'row_index': self.row_index,
            'column_name': self.column_name,
            'value': self.value,
            'detection_type': self.detection_type.value,
            'probability': self.probability,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'ml_features': self.ml_features
        }
    
    def is_high_probability(self, threshold: float = 0.8) -> bool:
        """Check if this is a high-probability detection."""
        return self.probability >= threshold
    
    def is_validation_error(self) -> bool:
        """Check if this is a validation error (highest confidence)."""
        return self.detection_type == DetectionType.VALIDATION


class UnifiedDetectorInterface(ABC):
    """
    Abstract base class for unified detectors that can combine multiple detection approaches.
    """
    
    def __init__(self, 
                 validator: Optional[Any] = None,
                 anomaly_detector: Optional[Any] = None,
                 ml_detector: Optional[Any] = None):
        """
        Initialize the unified detector with multiple detection components.
        
        Args:
            validator: Optional validator implementing ValidatorInterface
            anomaly_detector: Optional anomaly detector implementing AnomalyDetectorInterface
            ml_detector: Optional ML-based detector
        """
        self.validator = validator
        self.anomaly_detector = anomaly_detector
        self.ml_detector = ml_detector
    
    @abstractmethod
    def detect_issues(self, df: pd.DataFrame, column_name: str, 
                     enable_validation: bool = True,
                     enable_anomaly_detection: bool = True,
                     enable_ml_detection: bool = True,
                     validation_threshold: float = 0.0,
                     anomaly_threshold: float = 0.7,
                     ml_threshold: float = 0.7) -> List[UnifiedDetectionResult]:
        """
        Detect all types of issues in the specified column using enabled detection methods.
        
        Args:
            df: The dataframe to analyze
            column_name: The column to analyze
            enable_validation: Whether to run validation
            enable_anomaly_detection: Whether to run pattern-based anomaly detection
            enable_ml_detection: Whether to run ML-based anomaly detection
            validation_threshold: Minimum probability threshold for validation results
            anomaly_threshold: Minimum probability threshold for anomaly detection results
            ml_threshold: Minimum probability threshold for ML detection results
            
        Returns:
            List of UnifiedDetectionResult objects representing all detected issues
        """
        pass


class CombinedDetector(UnifiedDetectorInterface):
    """
    A concrete implementation that combines validation, anomaly detection, and ML detection.
    """
    
    def detect_issues(self, df: pd.DataFrame, column_name: str,
                     enable_validation: bool = True,
                     enable_anomaly_detection: bool = True,
                     enable_ml_detection: bool = True,
                     validation_threshold: float = 0.0,
                     anomaly_threshold: float = 0.7,
                     ml_threshold: float = 0.7) -> List[UnifiedDetectionResult]:
        """
        Detect issues using all available detection methods.
        """
        all_results = []
        
        # Run validation if enabled and available
        if enable_validation and self.validator:
            validation_errors = self.validator.bulk_validate(df, column_name)
            for error in validation_errors:
                if error.probability >= validation_threshold:
                    result = UnifiedDetectionResult.from_validation_error(error)
                    all_results.append(result)
        
        # Run pattern-based anomaly detection if enabled and available
        if enable_anomaly_detection and self.anomaly_detector:
            anomaly_errors = self.anomaly_detector.bulk_detect(df, column_name)
            for error in anomaly_errors:
                if error.probability >= anomaly_threshold:
                    result = UnifiedDetectionResult.from_anomaly_error(error)
                    all_results.append(result)
        
        # Run ML-based detection if enabled and available
        if enable_ml_detection and self.ml_detector:
            # This will need to be implemented based on your ML detector interface
            ml_results = self._run_ml_detection(df, column_name, ml_threshold)
            all_results.extend(ml_results)
        
        return all_results
    
    def _run_ml_detection(self, df: pd.DataFrame, column_name: str, 
                         threshold: float) -> List[UnifiedDetectionResult]:
        """
        Run ML-based detection using the configured ML detector.
        """
        ml_results = []
        
        if hasattr(self.ml_detector, 'bulk_detect'):
            # Use the ML detector's bulk_detect method
            try:
                ml_anomalies = self.ml_detector.bulk_detect(df, column_name)
                for ml_result in ml_anomalies:
                    # Convert MLAnomalyResult to UnifiedDetectionResult
                    unified_result = UnifiedDetectionResult.from_ml_anomaly_result(ml_result)
                    if unified_result.probability >= threshold:
                        ml_results.append(unified_result)
            except Exception as e:
                print(f"Warning: ML detection failed: {e}")
        
        return ml_results


class UnifiedReporter:
    """
    A reporter that can generate reports from unified detection results.
    """
    
    def __init__(self, include_technical_details: bool = False):
        """
        Initialize the unified reporter.
        
        Args:
            include_technical_details: Whether to include technical details in reports
        """
        self.include_technical_details = include_technical_details
    
    def generate_report(self, 
                       detection_results: List[UnifiedDetectionResult],
                       original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive report from unified detection results.
        
        Args:
            detection_results: List of UnifiedDetectionResult objects
            original_df: Original DataFrame for context
            
        Returns:
            List of report dictionaries
        """
        reports = []
        
        for result in detection_results:
            report = {
                'row_index': result.row_index,
                'column_name': result.column_name,
                'value': result.value,
                'detection_type': result.detection_type.value,
                'probability': result.probability,
                'error_code': result.error_code,
                'display_message': result.message,
                'is_high_probability': result.is_high_probability(),
                'is_validation_error': result.is_validation_error()
            }
            
            # Add technical details if requested
            if self.include_technical_details:
                report['details'] = result.details
                if result.ml_features:
                    report['ml_features'] = result.ml_features
            
            reports.append(report)
        
        return reports
    
    def generate_summary(self, detection_results: List[UnifiedDetectionResult]) -> Dict[str, Any]:
        """
        Generate a summary of detection results by type and probability.
        
        Args:
            detection_results: List of UnifiedDetectionResult objects
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_issues': len(detection_results),
            'by_type': {},
            'by_probability': {
                'high_probability': 0,
                'medium_probability': 0,
                'low_probability': 0
            },
            'validation_errors': 0,
            'anomalies': 0,
            'ml_anomalies': 0
        }
        
        for result in detection_results:
            # Count by type
            detection_type = result.detection_type.value
            summary['by_type'][detection_type] = summary['by_type'].get(detection_type, 0) + 1
            
            # Count by probability
            if result.probability >= 0.8:
                summary['by_probability']['high_probability'] += 1
            elif result.probability >= 0.5:
                summary['by_probability']['medium_probability'] += 1
            else:
                summary['by_probability']['low_probability'] += 1
            
            # Count by detection type
            if result.detection_type == DetectionType.VALIDATION:
                summary['validation_errors'] += 1
            elif result.detection_type == DetectionType.ANOMALY:
                summary['anomalies'] += 1
            elif result.detection_type == DetectionType.ML_ANOMALY:
                summary['ml_anomalies'] += 1
        
        return summary
