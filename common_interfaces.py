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
    
    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping.copy()
    
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
    
    @classmethod
    def from_default_mapping(cls) -> 'FieldMapper':
        return cls({
            "category": "article_structure_name_2",
            "color_name": "colour_name",
            "ean": "EAN",
            "article_number": "article_number",
            "colour_code": "colour_code",
            "customs_tariff_number": "customs_tariff_number",
            "description_short_1": "description_short_1",
            "long_description_nl": "long_description_NL",
            "material": "material",
            "product_name_en": "product_name_EN",
            "size": "size_name",
            "care_instructions": "Care Instructions",
            "season": "season",
            "manufactured_in": "Manufactured in",
            "supplier": "supplier",
            "brand": "brand",
            "collection": "collection",
        })


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