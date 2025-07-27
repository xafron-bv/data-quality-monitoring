"""
Common interfaces used across the data quality system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
from field_mapper import FieldMapper


class DetectionType(str, Enum):
    VALIDATION = "validation"
    ANOMALY = "anomaly" 
    ML_ANOMALY = "ml_anomaly"


@dataclass
class DetectionResult:
    """Unified result for all detection types."""
    row_index: int
    field_name: str
    value: Any
    detection_type: DetectionType
    confidence: float
    error_code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_high_confidence(self, threshold: float) -> bool:
        return self.confidence >= threshold


@dataclass
class AnomalyIssue:
    """Represents an anomaly detection issue."""
    detector_type: str
    confidence: float
    description: str
    suggested_action: Optional[str] = None
    details: Optional[Dict] = None


# Legacy compatibility functions
def convert_validation_error_to_detection_result(validation_error, field_name: str) -> DetectionResult:
    return DetectionResult(
        row_index=validation_error.row_index,
        field_name=field_name,
        value=validation_error.error_data,
        detection_type=DetectionType.VALIDATION,
        confidence=validation_error.probability,
        error_code=str(validation_error.error_type),
        message=f"Validation error: {validation_error.error_type}",
        details=validation_error.details or {}
    )


def convert_anomaly_error_to_detection_result(anomaly_error, field_name: str) -> DetectionResult:
    details = anomaly_error.details.copy() if anomaly_error.details else {}
    if hasattr(anomaly_error, 'explanation') and anomaly_error.explanation:
        details['explanation'] = anomaly_error.explanation
    
    return DetectionResult(
        row_index=anomaly_error.row_index,
        field_name=field_name,
        value=anomaly_error.anomaly_data,
        detection_type=DetectionType.ANOMALY,
        confidence=anomaly_error.probability,
        error_code=str(anomaly_error.anomaly_type),
        message=details.get('explanation', f"Anomaly detected: {anomaly_error.anomaly_type}"),
        details=details
    ) 