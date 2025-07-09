from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

from anomaly_detectors.anomaly_error import AnomalyError

class AnomalyReporterInterface(ABC):
    """
    Abstract Base Class for all Anomaly Reporter implementations.

    This interface ensures that every reporter can take a structured list of
    anomaly detection results and generate human-readable messages.
    """

    @abstractmethod
    def generate_report(self, anomaly_errors: List[AnomalyError], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of anomaly detection errors.

        Args:
            anomaly_errors (List[AnomalyError]): The list of AnomalyError objects
                produced by an AnomalyDetector.
            original_df (pd.DataFrame): The original DataFrame, useful for providing
                additional context in the report message.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a single reported anomaly and must contain at least:
            - 'row_index': The integer index of the row containing the anomaly.
            - 'error_data': The original anomalous data.
            - 'display_message': A human-readable string explaining the anomaly.
            - 'confidence': The confidence level of the anomaly detection.
            - 'anomaly': A boolean flag indicating this is an anomaly, not a validation error.
        """
        pass
