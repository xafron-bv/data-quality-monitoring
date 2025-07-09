from enum import Enum
from typing import Dict, Any, Optional, Union

class AnomalyError:
    """
    A standardized error class for anomaly detection.
    
    This class enforces that all anomaly results include:
    - An anomaly code/type (from a detector-specific Enum)
    - A confidence level (float between 0 and 1)
    - Optional details as a dictionary
    """
    
    def __init__(self, anomaly_type: Union[str, Enum], confidence: float, details: Optional[Dict[str, Any]] = None, 
                row_index: Optional[int] = None, column_name: Optional[str] = None, anomaly_data: Any = None):
        """
        Initialize an AnomalyError with required fields.
        
        Args:
            anomaly_type: The anomaly code/type (from detector-specific Enum)
            confidence: A value between 0 and 1 indicating the confidence level of the anomaly detection
            details: Optional dictionary containing additional anomaly details
            row_index: Optional row index where the anomaly was detected
            column_name: Optional column name where the anomaly was detected
            anomaly_data: Optional original data that caused the anomaly
        
        Raises:
            ValueError: If confidence is not between 0 and 1
        """
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        self.anomaly_type = anomaly_type
        self.confidence = confidence
        self.details = details or {}
        self.row_index = row_index
        self.column_name = column_name
        self.anomaly_data = anomaly_data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the anomaly to a dictionary format.
        
        Returns:
            Dict containing the anomaly information
        """
        result = {
            "anomaly_code": self.anomaly_type,
            "confidence": self.confidence,
            "details": self.details
        }
        
        if self.row_index is not None:
            result["row_index"] = self.row_index
            
        if self.column_name is not None:
            result["column_name"] = self.column_name
            
        if self.anomaly_data is not None:
            result["anomaly_data"] = self.anomaly_data
            
        return result
    
    def with_context(self, row_index: int, column_name: str, anomaly_data: Any) -> 'AnomalyError':
        """
        Creates a new AnomalyError with the same anomaly information but with added context
        
        Args:
            row_index: The index of the row where the anomaly was found
            column_name: The name of the column where the anomaly was found
            anomaly_data: The original data that caused the anomaly
            
        Returns:
            A new AnomalyError instance with the added context
        """
        return AnomalyError(
            anomaly_type=self.anomaly_type,
            confidence=self.confidence,
            details=self.details,
            row_index=row_index,
            column_name=column_name,
            anomaly_data=anomaly_data
        )
