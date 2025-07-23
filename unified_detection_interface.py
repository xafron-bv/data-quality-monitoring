"""Unified detection interface for data quality monitoring."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from validators.validation_error import ValidationError
from anomaly_detectors.anomaly_error import AnomalyError
from anomaly_detectors.reporter_interface import MLAnomalyResult
from common_interfaces import (
    DetectionResult, DetectionType, FieldMapper, 
    convert_validation_error_to_detection_result,
    convert_anomaly_error_to_detection_result
)
from exceptions import DataError


@dataclass
class DetectionConfig:
    """Configuration for detection operations."""
    validation_threshold: float
    anomaly_threshold: float
    ml_threshold: float
    enable_validation: bool = True
    enable_anomaly_detection: bool = True
    enable_ml_detection: bool = True


class UnifiedDetectorInterface(ABC):
    """Abstract base class for unified detectors."""
    
    def __init__(self, 
                 field_mapper: FieldMapper,
                 batch_size: Optional[int],
                 max_workers: int,
                 validator: Optional[Any] = None,
                 anomaly_detector: Optional[Any] = None,
                 ml_detector: Optional[Any] = None):
        self.field_mapper = field_mapper
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.validator = validator
        self.anomaly_detector = anomaly_detector
        self.ml_detector = ml_detector
    
    @abstractmethod
    def detect_issues(self, df: pd.DataFrame, field_name: str, 
                     config: DetectionConfig) -> List[DetectionResult]:
        pass


class CombinedDetector(UnifiedDetectorInterface):
    """Combines validation, anomaly detection, and ML detection."""
    
    def detect_issues(self, df: pd.DataFrame, field_name: str,
                     config: DetectionConfig) -> List[DetectionResult]:
        all_results = []
        
        try:
            column_name = self.field_mapper.validate_column_exists(df, field_name)
        except ValueError as e:
            raise DataError(str(e)) from e
        
        # Run validation if enabled
        if config.enable_validation and self.validator:
            validation_errors = self.validator.bulk_validate(df, column_name)
            for error in validation_errors:
                if error.probability >= config.validation_threshold:
                    result = convert_validation_error_to_detection_result(error, field_name)
                    all_results.append(result)
        
        # Run anomaly detection if enabled
        if config.enable_anomaly_detection and self.anomaly_detector:
            anomaly_errors = self.anomaly_detector.bulk_detect(df, column_name, self.batch_size, self.max_workers)
            for error in anomaly_errors:
                if error.probability >= config.anomaly_threshold:
                    result = convert_anomaly_error_to_detection_result(error, field_name)
                    all_results.append(result)
        
        # Run ML detection if enabled
        if config.enable_ml_detection and self.ml_detector:
            ml_results = self._run_ml_detection(df, field_name, column_name, config.ml_threshold)
            all_results.extend(ml_results)
        
        return all_results
    
    def _run_ml_detection(self, df: pd.DataFrame, field_name: str, column_name: str, 
                         threshold: float) -> List[DetectionResult]:
        ml_results = []
        
        if hasattr(self.ml_detector, 'bulk_detect'):
            try:
                ml_anomalies = self.ml_detector.bulk_detect(df, column_name, self.batch_size, self.max_workers)
                for ml_anomaly in ml_anomalies:
                    if ml_anomaly.probability >= threshold:
                        details = {}
                        if hasattr(ml_anomaly, 'explanation') and ml_anomaly.explanation:
                            details['explanation'] = ml_anomaly.explanation
                        
                        result = DetectionResult(
                            row_index=ml_anomaly.row_index,
                            field_name=field_name,
                            value=ml_anomaly.anomaly_data,
                            detection_type=DetectionType.ML_ANOMALY,
                            confidence=ml_anomaly.probability,
                            error_code="ML_ANOMALY",
                            message=details.get('explanation', f"ML anomaly detected with confidence {ml_anomaly.probability:.3f}"),
                            details=details
                        )
                        ml_results.append(result)
            except Exception as e:
                from exceptions import ModelError
                raise ModelError(
                    f"ML detection failed for field '{field_name}'",
                    details={'field_name': field_name, 'original_error': str(e)}
                ) from e
        
        return ml_results


class UnifiedReporter:
    """Reporter for unified detection results."""
    
    def __init__(self, high_confidence_threshold: float, include_technical_details: bool = False):
        self.include_technical_details = include_technical_details
        self.high_confidence_threshold = high_confidence_threshold
    
    def generate_report(self, 
                       detection_results: List[DetectionResult],
                       original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        reports = []
        
        for result in detection_results:
            report = {
                'row_index': result.row_index,
                'field_name': result.field_name,
                'value': result.value,
                'detection_type': result.detection_type.value,
                'confidence': result.confidence,
                'error_code': result.error_code,
                'display_message': result.message,
                'is_high_confidence': result.is_high_confidence(self.high_confidence_threshold)
            }
            
            if self.include_technical_details:
                report['details'] = result.details
            
            reports.append(report)
        
        return reports
