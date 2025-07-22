
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any, Optional, Dict
import concurrent.futures

from anomaly_detectors.anomaly_error import AnomalyError


def _process_batch_for_anomaly_detection(args):
    """
    Helper function for parallel processing of DataFrame batches.
    This function is defined at module level to be picklable by multiprocessing.
    
    Args:
        args: Tuple of (batch_df, column_name, detector_class, detector_args)
    """
    batch_df, column_name, detector_class, detector_args = args
    
    # Recreate the detector instance in the worker process
    detector_instance = detector_class(**detector_args)
    
    batch_anomalies = []
    for index, row in batch_df.iterrows():
        data = row[column_name]
        context = {col: row[col] for col in batch_df.columns}
        anomaly_error = detector_instance._detect_anomaly(data, context)
        if anomaly_error:
            error_with_context = anomaly_error.with_context(
                row_index=index,
                column_name=column_name,
                anomaly_data=data
            )
            batch_anomalies.append(error_with_context)
    return batch_anomalies

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

    def get_detector_args(self) -> Dict[str, Any]:
        """
        Return arguments needed to recreate this detector instance in a worker process.
        Subclasses should override this method to provide their initialization arguments.
        
        Returns:
            Dictionary of arguments that can be passed to the constructor
        """
        return {}

    def bulk_detect(self, df: pd.DataFrame, column_name: str, batch_size: int = None, max_workers: int = 7) -> List[AnomalyError]:
        """
        Detects anomalies in a column and returns a list of AnomalyError objects.
        This method is a non-editable engine that runs the `_detect_anomaly` logic in parallel batches.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
            column_name (str): The name of the column to check for anomalies.
            batch_size (int): Number of rows per batch for parallel processing. 
                             If None, automatically calculates optimal batch size based on data size and workers.
            max_workers (int): Number of parallel workers.

        Returns:
            List[AnomalyError]: A list of AnomalyError instances representing
            the anomalies found in the specified column.
        """
        anomalies = []

        # First learn patterns from the data (if the detector implements this)
        self.learn_patterns(df, column_name)

        # Calculate optimal batch size if not provided
        if batch_size is None:
            total_rows = len(df)
            # Aim for 1-2 batches per worker with larger minimum size for process efficiency
            optimal_batch_size = max(10000, min(100000, total_rows // max_workers))
            batch_size = optimal_batch_size

        # Split DataFrame into batches
        batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

        # Use ProcessPoolExecutor for true CPU parallelism
        # Prepare arguments for the worker function
        detector_class = self.__class__
        detector_args = self.get_detector_args()
        
        batch_args = [(batch, column_name, detector_class, detector_args) for batch in batches]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map batches to futures
            futures = [executor.submit(_process_batch_for_anomaly_detection, args) for args in batch_args]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                anomalies.extend(result)

        return anomalies
