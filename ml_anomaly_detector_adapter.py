"""
ML Anomaly Detector Adapter

This module provides an adapter that implements the unified detection interface
for ML-based anomaly detection using sentence transformers.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import sys

# Add the ml directory to the path to import ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'anomaly_detectors', 'ml'))

from unified_detection_interface import UnifiedDetectionResult, DetectionType
from anomaly_detectors.reporter_interface import MLAnomalyResult

try:
    from anomaly_detectors.ml.check_anomalies import (
        load_model_for_rule, 
        get_reference_clean_values_from_file,
        check_anomalies_with_references
    )
    from anomaly_detectors.ml.model_training import preprocess_text
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available: {e}")
    ML_AVAILABLE = False


class MLAnomalyDetectorAdapter:
    """
    Adapter that provides a unified interface to ML-based anomaly detection.
    This class wraps the existing ML detection functionality to work with the unified system.
    """
    
    def __init__(self, 
                 rule_name: str,
                 clean_data_path: str,
                 results_dir: str = 'anomaly_detectors/results',
                 max_clean_samples: int = 100):
        """
        Initialize the ML anomaly detector adapter.
        
        Args:
            rule_name: The name of the rule/model to use
            clean_data_path: Path to clean reference data CSV
            results_dir: Directory containing trained models
            max_clean_samples: Maximum number of clean samples to use for reference
        """
        self.rule_name = rule_name
        self.clean_data_path = clean_data_path
        self.results_dir = results_dir
        self.max_clean_samples = max_clean_samples
        self.model = None
        self.column_name = None
        self.clean_values = None
        
        if not ML_AVAILABLE:
            raise ImportError("ML dependencies not available. Please check your installation.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model and load reference data."""
        try:
            # Load the model for the specified rule
            self.model, self.column_name = load_model_for_rule(self.rule_name, self.results_dir)
            
            # Load reference clean values
            self.clean_values = get_reference_clean_values_from_file(
                self.clean_data_path, 
                self.column_name, 
                self.max_clean_samples
            )
            
            print(f"ML Detector initialized for rule '{self.rule_name}' on column '{self.column_name}'")
            print(f"Loaded {len(self.clean_values)} clean reference values")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ML detector: {e}")
    
    def detect_anomalies(self, 
                        df: pd.DataFrame, 
                        column_name: str,
                        threshold: float = 0.6) -> List[UnifiedDetectionResult]:
        """
        Detect anomalies in the specified column using the ML model.
        
        Args:
            df: DataFrame to analyze
            column_name: Column to analyze (should match the model's column)
            threshold: Anomaly threshold (lower values = more sensitive)
            
        Returns:
            List of UnifiedDetectionResult objects for detected anomalies
        """
        if column_name != self.column_name:
            raise ValueError(f"Column mismatch: model expects '{self.column_name}', got '{column_name}'")
        
        results = []
        
        # Extract values from the dataframe
        values = df[column_name].fillna('').astype(str).tolist()
        
        # Run anomaly detection
        anomaly_results = check_anomalies_with_references(
            self.model, 
            values, 
            self.clean_values, 
            threshold
        )
        
        # Convert results to unified format
        for i, (value, anomaly_info) in enumerate(zip(values, anomaly_results)):
            if anomaly_info['is_anomaly']:
                # Create ML anomaly result
                ml_result = MLAnomalyResult(
                    row_index=i,
                    column_name=column_name,
                    value=value,
                    anomaly_score=1.0 - anomaly_info['max_similarity'],  # Convert similarity to anomaly score
                    feature_contributions={'similarity_score': anomaly_info['max_similarity']},
                    nearest_neighbors=[(0, anomaly_info['max_similarity'])],  # Simplified
                    explanation=f"Low similarity to clean data (max similarity: {anomaly_info['max_similarity']:.3f})"
                )
                
                # Convert to unified result
                unified_result = UnifiedDetectionResult.from_ml_anomaly_result(ml_result)
                results.append(unified_result)
        
        return results
    
    def bulk_detect(self, df: pd.DataFrame, column_name: str) -> List[MLAnomalyResult]:
        """
        Detect anomalies and return raw ML results (for compatibility with existing interfaces).
        
        Args:
            df: DataFrame to analyze
            column_name: Column to analyze
            
        Returns:
            List of MLAnomalyResult objects
        """
        unified_results = self.detect_anomalies(df, column_name)
        
        ml_results = []
        for unified_result in unified_results:
            if unified_result.detection_type == DetectionType.ML_ANOMALY:
                # Reconstruct MLAnomalyResult from unified result
                ml_result = MLAnomalyResult(
                    row_index=unified_result.row_index,
                    column_name=unified_result.column_name,
                    value=unified_result.value,
                    anomaly_score=unified_result.confidence,
                    feature_contributions=unified_result.ml_features.get('feature_contributions', {}),
                    nearest_neighbors=unified_result.ml_features.get('nearest_neighbors', []),
                    cluster_info=unified_result.ml_features.get('cluster_info', {}),
                    probability_info=unified_result.ml_features.get('probability_info', {}),
                    explanation=unified_result.details.get('explanation')
                )
                ml_results.append(ml_result)
        
        return ml_results


class MLDetectorFactory:
    """
    Factory class for creating ML detector adapters for different rules/columns.
    """
    
    def __init__(self, 
                 clean_data_path: str,
                 results_dir: str = 'anomaly_detectors/results'):
        """
        Initialize the factory.
        
        Args:
            clean_data_path: Path to clean reference data CSV
            results_dir: Directory containing trained models
        """
        self.clean_data_path = clean_data_path
        self.results_dir = results_dir
        self._detectors = {}
    
    def get_detector(self, rule_name: str) -> MLAnomalyDetectorAdapter:
        """
        Get or create an ML detector for the specified rule.
        
        Args:
            rule_name: The name of the rule/model
            
        Returns:
            MLAnomalyDetectorAdapter instance
        """
        if rule_name not in self._detectors:
            self._detectors[rule_name] = MLAnomalyDetectorAdapter(
                rule_name=rule_name,
                clean_data_path=self.clean_data_path,
                results_dir=self.results_dir
            )
        
        return self._detectors[rule_name]
    
    def list_available_rules(self) -> List[str]:
        """
        List available rules/models in the results directory.
        
        Returns:
            List of available rule names
        """
        try:
            from anomaly_detectors.ml.rule_column_map import get_rule_to_column_map
            rule_map = get_rule_to_column_map()
            
            available_rules = []
            for rule_name, column_name in rule_map.items():
                model_dir = os.path.join(
                    self.results_dir, 
                    f'results_{column_name.replace(" ", "_").lower()}'
                )
                if os.path.isdir(model_dir):
                    available_rules.append(rule_name)
            
            return available_rules
            
        except Exception as e:
            print(f"Warning: Could not list available rules: {e}")
            return []


# Example usage functions
def create_ml_detector_for_column(column_name: str, 
                                 clean_data_path: str,
                                 rule_name: str = None) -> MLAnomalyDetectorAdapter:
    """
    Convenience function to create an ML detector for a specific column.
    
    Args:
        column_name: The column to detect anomalies in
        clean_data_path: Path to clean reference data
        rule_name: Optional specific rule name (defaults to column_name)
        
    Returns:
        MLAnomalyDetectorAdapter instance
    """
    if rule_name is None:
        rule_name = column_name.lower().replace(' ', '_')
    
    return MLAnomalyDetectorAdapter(
        rule_name=rule_name,
        clean_data_path=clean_data_path
    )


def test_ml_detector(detector: MLAnomalyDetectorAdapter, 
                    test_data: pd.DataFrame,
                    column_name: str,
                    threshold: float = 0.6) -> Dict[str, Any]:
    """
    Test an ML detector and return summary statistics.
    
    Args:
        detector: The ML detector to test
        test_data: DataFrame containing test data
        column_name: Column to test
        threshold: Anomaly detection threshold
        
    Returns:
        Dictionary containing test results and statistics
    """
    results = detector.detect_anomalies(test_data, column_name, threshold)
    
    total_rows = len(test_data)
    anomalies_found = len(results)
    anomaly_rate = anomalies_found / total_rows if total_rows > 0 else 0
    
    confidence_scores = [r.confidence for r in results]
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    return {
        'total_rows': total_rows,
        'anomalies_found': anomalies_found,
        'anomaly_rate': anomaly_rate,
        'average_confidence': avg_confidence,
        'threshold_used': threshold,
        'rule_name': detector.rule_name,
        'column_name': detector.column_name
    }
