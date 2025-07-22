from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union, Tuple

from anomaly_detectors.anomaly_error import AnomalyError

class MLAnomalyResult:
    """
    Represents the result of an ML-based anomaly detection model.
    This provides more detailed information than the simpler AnomalyError.
    """
    
    def __init__(self, 
                row_index: int, 
                column_name: str,
                value: Any,
                probabiliy: float,
                feature_contributions: Dict[str, float] = None,
                nearest_neighbors: List[Tuple[int, float]] = None,
                cluster_info: Dict[str, Any] = None,
                probability_info: Dict[str, Any] = None,
                explanation: str = None):
        """
        Initialize a machine learning anomaly result with detailed information.
        
        Args:
            row_index: The index of the row where the anomaly was found
            column_name: The name of the column where the anomaly was found
            value: The original data value that was flagged as anomalous
            probabiliy: A value between 0 and 1 indicating the anomaly score
            feature_contributions: Dictionary showing how much each feature contributed to the score
            nearest_neighbors: List of (row_index, distance) tuples to nearest normal data points
            cluster_info: Information about clustering results (if applicable)
            probability_info: Statistical information about probabilities and distributions
            explanation: Optional model-generated explanation of why this is anomalous
        """
        self.row_index = row_index
        self.column_name = column_name
        self.value = value
        self.probabiliy = probabiliy
        self.feature_contributions = feature_contributions or {}
        self.nearest_neighbors = nearest_neighbors or []
        self.cluster_info = cluster_info or {}
        self.probability_info = probability_info or {}
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ML anomaly result to a dictionary format."""
        return {
            "row_index": self.row_index,
            "column_name": self.column_name,
            "value": self.value,
            "probabiliy": self.probabiliy,
            "feature_contributions": self.feature_contributions,
            "nearest_neighbors": self.nearest_neighbors,
            "cluster_info": self.cluster_info,
            "probability_info": self.probability_info,
            "explanation": self.explanation
        }

class AnomalyReporterInterface(ABC):
    """
    Abstract Base Class for all Anomaly Reporter implementations.

    This interface ensures that every reporter can take a structured list of
    anomaly detection results and generate human-readable messages. It supports
    both rule-based anomaly errors and ML-based anomaly results.
    """

    @abstractmethod
    def generate_report(self, 
                       anomaly_results: Union[List[AnomalyError], List[MLAnomalyResult]], 
                       original_df: pd.DataFrame,
                       threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of anomaly detection results.

        Args:
            anomaly_results: Either a list of AnomalyError objects (for rule-based detection)
                or a list of MLAnomalyResult objects (for ML-based detection).
            original_df: The original DataFrame, useful for providing
                additional context in the report message.
            threshold: For ML-based results, only report anomalies with score above this threshold.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a single reported anomaly and must contain at least:
            - 'row_index': The integer index of the row containing the anomaly.
            - 'value': The original anomalous data value.
            - 'display_message': A human-readable string explaining the anomaly.
            - 'probabiliy': The probability level or score of the anomaly detection.
            - 'explanation': Detailed explanation of why this was flagged as anomalous.
            - 'feature_importance': For ML models, which features contributed to the detection.
            - 'is_ml_based': Boolean indicating if this came from an ML model or rule-based system.
        """
        pass
