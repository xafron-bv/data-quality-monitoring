from enum import Enum
from typing import Dict, Any, Optional, Union, List, Tuple

class AnomalyError:
    """
    A standardized error class for anomaly detection.
    
    This class enforces that all anomaly results include:
    - An anomaly code/type (from a detector-specific Enum)
    - An anomaly score (float between 0 and 1)
    - Optional details as a dictionary
    
    It also supports ML-style outputs including:
    - Feature contributions
    - Nearest neighbors information
    - Cluster information
    - Probability distribution information
    - Natural language explanation
    """
    
    def __init__(self, 
                anomaly_type: Union[str, Enum], 
                anomaly_score: float, 
                details: Optional[Dict[str, Any]] = None, 
                row_index: Optional[int] = None, 
                column_name: Optional[str] = None, 
                anomaly_data: Any = None,
                feature_contributions: Optional[Dict[str, float]] = None,
                nearest_neighbors: Optional[List[Tuple[int, float]]] = None,
                cluster_info: Optional[Dict[str, Any]] = None,
                probability_info: Optional[Dict[str, Any]] = None,
                explanation: Optional[str] = None):
        """
        Initialize an AnomalyError with required fields.
        
        Args:
            anomaly_type: The anomaly code/type (from detector-specific Enum)
            anomaly_score: A value between 0 and 1 indicating the anomaly severity
            details: Optional dictionary containing additional anomaly details
            row_index: Optional row index where the anomaly was detected
            column_name: Optional column name where the anomaly was detected
            anomaly_data: Optional original data that caused the anomaly
            feature_contributions: Dictionary showing how much each feature contributed to the score
            nearest_neighbors: List of (row_index, distance) tuples to nearest normal data points
            cluster_info: Information about clustering results (if applicable)
            probability_info: Statistical information about probabilities and distributions
            explanation: Natural language explanation of why this is anomalous
        
        Raises:
            ValueError: If anomaly_score is not between 0 and 1
        """
        if not 0 <= anomaly_score <= 1:
            raise ValueError(f"Anomaly score must be between 0 and 1, got {anomaly_score}")
        
        self.anomaly_type = anomaly_type
        self.anomaly_score = anomaly_score
        self.details = details or {}
        self.row_index = row_index
        self.column_name = column_name
        self.anomaly_data = anomaly_data
        self.feature_contributions = feature_contributions or {}
        self.nearest_neighbors = nearest_neighbors or []
        self.cluster_info = cluster_info or {}
        self.probability_info = probability_info or {}
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the anomaly to a dictionary format.
        
        Returns:
            Dict containing the anomaly information
        """
        result = {
            "anomaly_code": self.anomaly_type,
            "anomaly_score": self.anomaly_score,
            "details": self.details,
            "feature_contributions": self.feature_contributions,
            "nearest_neighbors": self.nearest_neighbors,
            "cluster_info": self.cluster_info,
            "probability_info": self.probability_info,
            "explanation": self.explanation
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
            anomaly_score=self.anomaly_score,
            details=self.details,
            row_index=row_index,
            column_name=column_name,
            anomaly_data=anomaly_data,
            feature_contributions=self.feature_contributions,
            nearest_neighbors=self.nearest_neighbors,
            cluster_info=self.cluster_info,
            probability_info=self.probability_info,
            explanation=self.explanation
        )
