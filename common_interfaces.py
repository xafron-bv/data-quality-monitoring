"""Unified interfaces for data quality monitoring."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd


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


class FieldMapper:
    """Centralized field-to-column mapping service."""
    
    def __init__(self, mapping: Dict[str, str], brand_name: Optional[str] = None):
        self._mapping = mapping.copy()
        self._brand_name = brand_name
    
    def get_column_name(self, field_name: str) -> str:
        return self._mapping.get(field_name, field_name)
    
    def validate_column_exists(self, df: pd.DataFrame, field_name: str) -> str:
        column_name = self.get_column_name(field_name)
        if column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' (mapped from field '{field_name}') "
                f"not found in DataFrame. Available columns: {list(df.columns)}"
            )
        return column_name
    
    def get_available_fields(self) -> List[str]:
        return list(self._mapping.keys())
    
    def get_brand_name(self) -> Optional[str]:
        """Get the brand name associated with this mapper."""
        return self._brand_name
    
    @classmethod
    def from_default_mapping(cls) -> 'FieldMapper':
        """Get field mapper for current brand."""
        from brand_configs import get_brand_config_manager
        manager = get_brand_config_manager()
        current_brand = manager.get_current_brand()
        if not current_brand:
            raise ValueError("No brand configured. Please specify a brand or create a brand configuration.")
        return cls(current_brand.field_mappings, current_brand.brand_name)
    
    @classmethod
    def from_brand(cls, brand_name: str) -> 'FieldMapper':
        """Get field mapper for a specific brand."""
        from brand_configs import get_brand_config_manager
        manager = get_brand_config_manager()
        return manager.get_field_mapper(brand_name)


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