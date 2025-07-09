from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any, Optional, Dict

from anomaly_detectors.anomaly_error import AnomalyError

class AnomalyDetectorInterface(ABC):
    """
    Abstract Base Class for all Anomaly Detector implementations.

    This interface guarantees that every anomaly detector provides a consistent way
    for the orchestration system to initiate an anomaly detection process on a dataset.
    """

    @abstractmethod
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Contains the specific anomaly detection logic for a single data entry.
        This method must be implemented by subclasses.

        Args:
            value: The data from the DataFrame column to be checked for anomalies.
            context: Optional dictionary containing additional context data for anomaly detection.

        Returns:
            - None if no anomaly is detected.
            - An AnomalyError instance if an anomaly is detected.
        """
        pass

    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learns normal patterns from the data to establish a baseline for anomaly detection.
        This is an optional method that anomaly detectors can override to learn from the data.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to learn from.
            column_name (str): The name of the column to learn patterns from.
        """
        pass

    def bulk_detect(self, df: pd.DataFrame, column_name: str) -> List[AnomalyError]:
        """
        Detects anomalies in a column and returns a list of AnomalyError objects.
        This method is a non-editable engine that runs the `_detect_anomaly` logic.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
            column_name (str): The name of the column to check for anomalies.

        Returns:
            List[AnomalyError]: A list of AnomalyError instances representing
            the anomalies found in the specified column.
        """
        anomalies = []

        # First learn patterns from the data (if the detector implements this)
        self.learn_patterns(df, column_name)
        
        # Then detect anomalies in each row
        for index, row in df.iterrows():
            data = row[column_name]
            
            # Build context from the row data for contextual anomaly detection
            context = {col: row[col] for col in df.columns}
            
            # The implemented logic in _detect_anomaly is called for every row.
            anomaly_error = self._detect_anomaly(data, context)

            # If the custom logic returned an error, add context and add it to the list
            if anomaly_error:
                # Add row and column context to the anomaly error
                error_with_context = anomaly_error.with_context(
                    row_index=index,
                    column_name=column_name,
                    anomaly_data=data
                )
                anomalies.append(error_with_context)
                
        return anomalies
