
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any, Optional, Dict
import concurrent.futures

from anomaly_detectors.anomaly_error import AnomalyError
from anomaly_detectors.ml_based.gpu_utils import get_optimal_batch_size
from anomaly_detectors.ml_based.check_anomalies import check_anomalies
        
from common.debug_config import debug_print


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
    
    # For ML detectors with GPU support, process in larger sub-batches for efficiency
    if hasattr(detector_instance, 'use_gpu') and detector_instance.use_gpu:
        # Process the entire batch at once for GPU efficiency
        return _process_ml_batch_gpu(batch_df, column_name, detector_instance)
    else:
        # Standard row-by-row processing for CPU-based detectors
        return _process_batch_cpu(batch_df, column_name, detector_instance)


def _process_ml_batch_gpu(batch_df, column_name, detector_instance):
    """GPU-optimized batch processing for ML detectors."""
    batch_anomalies = []
    
    # Extract all values for batch processing
    values = batch_df[column_name].tolist()
    indices = batch_df.index.tolist()
    
    # Process all values at once using GPU acceleration
    try:
        # Load patterns if not already done
        if not detector_instance.is_initialized:
            detector_instance.learn_patterns(batch_df, column_name)
        
        # Check anomalies for the entire batch
        results = check_anomalies(detector_instance.model, values, detector_instance.threshold)
        
        # Convert results to AnomalyError objects
        for i, (index, result) in enumerate(zip(indices, results)):
            if result['is_anomaly']:
                # Create context from the row data
                row = batch_df.loc[index]
                context = {col: row[col] for col in batch_df.columns}
                
                anomaly_error = detector_instance._detect_anomaly(result['value'], context)
                if anomaly_error:
                    error_with_context = anomaly_error.with_context(
                        row_index=index,
                        column_name=column_name,
                        anomaly_data=result['value']
                    )
                    batch_anomalies.append(error_with_context)
    
    except Exception as e:
        print(f"GPU batch processing failed, falling back to CPU: {e}")
        # Fallback to standard processing
        return _process_batch_cpu(batch_df, column_name, detector_instance)
    
    return batch_anomalies


def _process_batch_cpu(batch_df, column_name, detector_instance):
    """Standard CPU batch processing."""
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

    def bulk_detect(self, df: pd.DataFrame, column_name: str, batch_size: Optional[int], max_workers: int) -> List[AnomalyError]:
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

        # NOTE: Pattern learning should NOT happen during detection!
        # Pattern-based detectors should either:
        # 1. Use fixed rules that don't require learning, OR
        # 2. Be pre-trained separately before detection
        # Only ML-based detectors require separate training via their training pipeline

        # Calculate optimal batch size if not provided
        if batch_size is None:
            # Use GPU utilities to determine optimal batch size
            if hasattr(self, 'use_gpu') and self.use_gpu:
                # For ML detectors with GPU, use GPU-optimized batch size
                batch_size = get_optimal_batch_size('cuda')
            else:
                # For CPU detectors, use CPU-optimized batch size
                batch_size = get_optimal_batch_size('cpu')

        # Split DataFrame into batches
        batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
        
        detector_type = "GPU ML" if (hasattr(self, 'use_gpu') and self.use_gpu) else "CPU"
        detector_name = self.__class__.__name__
        print(f"[{detector_name} - {detector_type}] Processing {len(df)} rows with batch size {batch_size}, creating {len(batches)} batches")
        for i, batch in enumerate(batches):
            print(f"  Batch {i+1}: {len(batch)} rows (indices {batch.index.min()}-{batch.index.max()})")

        # Use ProcessPoolExecutor for true CPU parallelism
        # Prepare arguments for the worker function
        detector_class = self.__class__
        detector_args = self.get_detector_args()
        
        batch_args = [(batch, column_name, detector_class, detector_args) for batch in batches]
        
        debug_print(f"[DEBUG] Created {len(batch_args)} batch_args for {len(batches)} batches")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map batches to futures
            futures = [executor.submit(_process_batch_for_anomaly_detection, args) for args in batch_args]
            debug_print(f"[DEBUG] Submitted {len(futures)} futures to executor")
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                debug_print(f"[DEBUG] Processing future {i+1}/{len(futures)}")
                result = future.result()
                anomalies.extend(result)

        return anomalies
