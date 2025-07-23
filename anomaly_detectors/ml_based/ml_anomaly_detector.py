"""
ML-based Anomaly Detector

This module implements the AnomalyDetectorInterface for ML-based anomaly detection
using sentence transformers and trained models.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import os
import sys

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError

# Add the ml_based directory to the path to import ML modules
sys.path.append(os.path.dirname(__file__))

try:
    from check_anomalies import (
        load_model_for_field,
        check_anomalies
    )
    from model_training import preprocess_text
    from field_column_map import get_field_to_column_map
    from gpu_utils import get_optimal_device, print_device_info
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available: {e}")
    ML_AVAILABLE = False


class MLAnomalyDetector(AnomalyDetectorInterface):
    """
    ML-based anomaly detector that implements the AnomalyDetectorInterface.
    This detector uses trained sentence transformer models to detect anomalies
    using centroid-based distance calculations.
    """
    
    # Class-level cache to share models across instances with same parameters
    _model_cache = {}
    
    def __init__(self, 
                 field_name: str,
                 threshold: float,
                 results_dir: str = None,
                 use_gpu: bool = True):
        """
        Initialize the ML anomaly detector.
        
        Args:
            field_name: The type of field to validate
            results_dir: Directory containing trained models (defaults to ml_based/results)
            threshold: Similarity threshold for anomaly detection (lower = more sensitive)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.field_name = field_name
        if results_dir is None:
            # Default to the results directory in the parent anomaly_detectors folder
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        self.results_dir = results_dir
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.model = None
        self.column_name = None
        self.is_initialized = False
        
        if not ML_AVAILABLE:
            raise ImportError("ML dependencies not available. Please check your installation.")
    
    def _get_cache_key(self):
        """Generate a cache key for this detector configuration."""
        return (self.field_name, self.results_dir, self.use_gpu)
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn patterns by loading the trained model (with caching).
        
        Args:
            df: DataFrame containing the data (not used, patterns are pre-learned)
            column_name: Column name to detect anomalies in
        """
        if self.is_initialized:
            return
        
        # Check class-level cache first
        cache_key = self._get_cache_key()
        if cache_key in MLAnomalyDetector._model_cache:
            self.model, self.column_name = MLAnomalyDetector._model_cache[cache_key]
            print(f"Using cached model for field '{self.field_name}' on column '{self.column_name}'")
        else:
            try:
                # Load the model for the specified field with GPU support
                self.model, self.column_name = load_model_for_field(self.field_name, self.results_dir, self.use_gpu)
                
                # Cache the loaded model
                MLAnomalyDetector._model_cache[cache_key] = (self.model, self.column_name)
                print(f"ML Detector initialized for field '{self.field_name}' on column '{self.column_name}'")
            
            except Exception as e:
                print(f"Error loading ML model for field '{self.field_name}': {e}")
                self.model = None
                self.column_name = None
                self.is_initialized = False
                return
        
        # Verify column name matches
        if column_name != self.column_name:
            print(f"Warning: Column mismatch - expected '{self.column_name}', got '{column_name}'")
            # Update column name to match the requested one
            self.column_name = column_name
        
        self.is_initialized = True
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomaly in a single value using the ML model.
        
        Args:
            value: The value to check for anomaly
            context: Additional context (not used in this implementation)
            
        Returns:
            AnomalyError if an anomaly is detected, None otherwise
        """
        if not self.is_initialized:
            return None
            
        try:
            # Preprocess the value
            processed_value = preprocess_text(value)
            if processed_value is None:
                processed_value = ""
            
            # Run anomaly detection on single value using centroid-based approach
            anomaly_results = check_anomalies(
                self.model, 
                [processed_value], 
                self.threshold
            )
            
            if anomaly_results and len(anomaly_results) > 0:
                result = anomaly_results[0]
                
                if result['is_anomaly']:
                    # Get similarity score (probability_of_correctness)
                    similarity_score = result.get('probability_of_correctness', 0)
                    anomaly_probability = 1.0 - similarity_score
                    
                    # Create AnomalyError with ML-specific information
                    return AnomalyError(
                        anomaly_type="ML_ANOMALY",
                        probability=anomaly_probability,
                        details={
                            'similarity_score': similarity_score,
                            'threshold_used': self.threshold,
                            'field_name': self.field_name,
                            'model_type': 'sentence_transformer'
                        },
                        feature_contributions={'similarity_score': similarity_score},
                        explanation=f"Low similarity to data centroid (similarity: {similarity_score:.3f})"
                    )
            
            return None
            
        except Exception as e:
            print(f"Error in ML anomaly detection: {e}")
            return None
    
    def get_detector_args(self) -> Dict[str, Any]:
        """
        Return arguments needed to recreate this detector instance in a worker process.
        
        Returns:
            Dictionary of arguments that can be passed to the constructor
        """
        return {
            'field_name': self.field_name,
            'results_dir': self.results_dir,
            'threshold': self.threshold,
            'use_gpu': self.use_gpu
        }


class MLAnomalyDetectorFactory:
    """
    Factory class for creating ML anomaly detectors for different field types.
    """
    
    def __init__(self, 
                 results_dir: str = None,
                 threshold: float = 0.6):
        """
        Initialize the factory.
        
        Args:
            results_dir: Directory containing trained models (defaults to anomaly_detectors/results)
            threshold: Default threshold for anomaly detection
        """
        if results_dir is None:
            # Default to the results directory in the parent anomaly_detectors folder
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        self.results_dir = results_dir
        self.threshold = threshold
        self._detectors = {}
    
    def get_detector(self, field_name: str) -> MLAnomalyDetector:
        """
        Get or create an ML detector for the specified field.
        
        Args:
            field_name: The type of field to validate
            
        Returns:
            MLAnomalyDetector instance
        """
        if field_name not in self._detectors:
            self._detectors[field_name] = MLAnomalyDetector(
                field_name=field_name,
                results_dir=self.results_dir,
                threshold=self.threshold
            )
        
        return self._detectors[field_name]
    
    def list_available_fields(self) -> List[str]:
        """
        List available field types/models in the results directory.
        
        Returns:
            List of available field names
        """
        try:
            field_map = get_field_to_column_map()
            
            available_fields = []
            for field_name, column_name in field_map.items():
                model_dir = os.path.join(
                    self.results_dir, 
                    f'results_{column_name.replace(" ", "_").lower()}'
                )
                if os.path.isdir(model_dir):
                    available_fields.append(field_name)
            
            return available_fields
            
        except Exception as e:
            print(f"Warning: Could not list available fields: {e}")
            return []


# Convenience functions for creating ML detectors
def create_ml_detector_for_field(field_name: str, 
                               results_dir: str = None,
                               threshold: float = 0.6) -> MLAnomalyDetector:
    """
    Convenience function to create an ML detector for a specific field.
    
    Args:
        field_name: The field name to create detector for
        results_dir: Directory containing trained models (defaults to anomaly_detectors/results)
        threshold: Anomaly detection threshold
        
    Returns:
        MLAnomalyDetector instance
    """
    if results_dir is None:
        # Default to the results directory in the parent anomaly_detectors folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    return MLAnomalyDetector(
        field_name=field_name,
        results_dir=results_dir,
        threshold=threshold
    )


def create_ml_detector_for_column(column_name: str,
                                 results_dir: str = None,
                                 threshold: float = 0.6) -> Optional[MLAnomalyDetector]:
    """
    Convenience function to create an ML detector for a specific column.
    
    Args:
        column_name: The column to detect anomalies in
        results_dir: Directory containing trained models (defaults to anomaly_detectors/results)
        threshold: Anomaly detection threshold
        
    Returns:
        MLAnomalyDetector instance or None if no field found for column
    """
    if results_dir is None:
        # Default to the results directory in the parent anomaly_detectors folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    try:
        field_map = get_field_to_column_map()
        
        # Find field name for this column
        field_name = None
        for field, col in field_map.items():
            if col == column_name:
                field_name = field
                break
        
        if field_name is None:
            print(f"No field found for column '{column_name}'")
            return None
        
        return MLAnomalyDetector(
            field_name=field_name,
            results_dir=results_dir,
            threshold=threshold
        )
        
    except Exception as e:
        print(f"Error creating ML detector for column '{column_name}': {e}")
        return None
