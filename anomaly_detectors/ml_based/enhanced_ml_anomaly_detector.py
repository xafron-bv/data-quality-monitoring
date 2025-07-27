"""
Enhanced ML Anomaly Detector with AnomalyLLM Integration

This module extends the existing ML anomaly detector with AnomalyLLM concepts:
- Few-shot learning capabilities
- Dynamic-aware encoding
- In-context learning
- Prototype-based reprogramming
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import os
import sys
import numpy as np
from datetime import datetime

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError

# Add the ml_based directory to the path to import ML modules
sys.path.append(os.path.dirname(__file__))

try:
    from anomaly_detectors.ml_based.anomalyllm_integration import (
        AnomalyLLMIntegration, FewShotExample, DynamicContext
    )
    from anomaly_detectors.ml_based.check_anomalies import (
        load_model_for_field
    )
    from anomaly_detectors.ml_based.model_training import preprocess_text
    from field_column_map import get_field_to_column_map
    from anomaly_detectors.ml_based.gpu_utils import get_optimal_device, print_device_info
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Error: Enhanced ML modules failed to import: {e}")
    ML_AVAILABLE = False


class EnhancedMLAnomalyDetector(AnomalyDetectorInterface):
    """
    Enhanced ML-based anomaly detector that integrates AnomalyLLM concepts
    with the existing ML anomaly detection framework.
    
    Key enhancements:
    - Few-shot learning with in-context examples
    - Dynamic-aware encoding for temporal patterns
    - Prototype-based reprogramming
    - Enhanced explanations and context awareness
    """
    
    # Class-level cache to share models across instances
    _model_cache = {}
    
    def __init__(self, 
                 field_name: str,
                 threshold: float,
                 results_dir: str = None,
                 use_gpu: bool = True,
                 enable_anomalyllm: bool = True,
                 few_shot_examples: Optional[List[FewShotExample]] = None,
                 temporal_column: Optional[str] = None,
                 context_columns: Optional[List[str]] = None):
        """
        Initialize the enhanced ML anomaly detector.
        
        Args:
            field_name: The type of field to validate
            results_dir: Directory containing trained models
            threshold: Similarity threshold for anomaly detection
            use_gpu: Whether to use GPU acceleration
            enable_anomalyllm: Whether to enable AnomalyLLM features
            few_shot_examples: Optional few-shot examples for in-context learning
            temporal_column: Optional column containing temporal information
            context_columns: Optional columns containing contextual information
        """
        self.field_name = field_name
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        self.results_dir = results_dir
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.enable_anomalyllm = enable_anomalyllm
        self.few_shot_examples = few_shot_examples or []
        self.temporal_column = temporal_column
        self.context_columns = context_columns or []
        
        # Model components
        self.base_model = None
        self.anomalyllm_integration = None
        self.column_name = None
        self.reference_centroid = None
        self.is_initialized = False
        
        if not ML_AVAILABLE:
            raise ImportError("ML dependencies not available. Please check your installation.")
    
    def _get_cache_key(self):
        """Generate a cache key for this detector configuration."""
        return (
            self.field_name, 
            self.results_dir, 
            self.use_gpu, 
            self.enable_anomalyllm,
            tuple(sorted(self.context_columns)) if self.context_columns else None
        )
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn patterns by loading the trained model and setting up AnomalyLLM integration.
        
        Args:
            df: DataFrame containing the data
            column_name: Column name to detect anomalies in
        """
        if self.is_initialized:
            return
        
        # Check class-level cache first
        cache_key = self._get_cache_key()
        if cache_key in EnhancedMLAnomalyDetector._model_cache:
            cached_data = EnhancedMLAnomalyDetector._model_cache[cache_key]
            self.base_model, self.anomalyllm_integration, self.column_name, self.reference_centroid = cached_data
            print(f"Using cached enhanced model for field '{self.field_name}' on column '{self.column_name}'")
        else:
            try:
                # Load the base model and reference centroid
                self.base_model, self.column_name, self.reference_centroid = load_model_for_field(
                    self.field_name, self.results_dir, self.use_gpu
                )
                
                # Initialize AnomalyLLM integration if enabled
                if self.enable_anomalyllm:
                    self.anomalyllm_integration = AnomalyLLMIntegration(
                        self.base_model,
                        enable_dynamic_encoding=True,
                        enable_prototype_reprogramming=True,
                        enable_in_context_learning=True
                    )
                    
                    # Train with AnomalyLLM concepts if we have the required data
                    self._train_with_anomalyllm_concepts(df, column_name)
                
                # Cache the loaded components
                cached_data = (self.base_model, self.anomalyllm_integration, self.column_name, self.reference_centroid)
                EnhancedMLAnomalyDetector._model_cache[cache_key] = cached_data
                print(f"Enhanced ML Detector initialized for field '{self.field_name}' on column '{self.column_name}'")
                
            except Exception as e:
                print(f"Error loading enhanced ML model for field '{self.field_name}': {e}")
                self.base_model = None
                self.anomalyllm_integration = None
                self.column_name = None
                self.reference_centroid = None
                self.is_initialized = False
                return
        
        # Verify column name matches
        if column_name != self.column_name:
            print(f"Warning: Column mismatch - expected '{self.column_name}', got '{column_name}'")
            self.column_name = column_name
        
        self.is_initialized = True
    
    def _train_with_anomalyllm_concepts(self, df: pd.DataFrame, column_name: str):
        """Train the model with AnomalyLLM concepts."""
        if not self.anomalyllm_integration:
            return
        
        try:
            # Prepare few-shot examples if available
            few_shot_examples = self._prepare_few_shot_examples(df, column_name)
            
            # Train with AnomalyLLM concepts
            training_results = self.anomalyllm_integration.train_with_anomalyllm_concepts(
                df=df,
                column_name=column_name,
                field_name=self.field_name,
                few_shot_examples=few_shot_examples,
                temporal_column=self.temporal_column,
                context_columns=self.context_columns
            )
            
            print(f"✅ AnomalyLLM training completed:")
            print(f"   - Dynamic encoding: {training_results.get('dynamic_encoder_trained', False)}")
            print(f"   - Prototypes learned: {training_results.get('prototypes_learned', False)}")
            print(f"   - In-context learning: {training_results.get('in_context_learning_setup', False)}")
            if training_results.get('num_few_shot_examples'):
                print(f"   - Few-shot examples: {training_results['num_few_shot_examples']}")
            
        except Exception as e:
            print(f"⚠️  AnomalyLLM training failed: {e}")
            # Continue with standard detection
    
    def _prepare_few_shot_examples(self, df: pd.DataFrame, column_name: str) -> List[FewShotExample]:
        """Prepare few-shot examples from the data."""
        examples = []
        
        # Add user-provided examples
        examples.extend(self.few_shot_examples)
        
        # Generate additional examples from the data if we have few examples
        if len(examples) < 5:
            # Sample some normal values as examples
            clean_values = df[column_name].dropna().unique()
            sample_size = min(3, len(clean_values))
            
            for i, value in enumerate(clean_values[:sample_size]):
                if str(value).strip():
                    examples.append(FewShotExample(
                        value=str(value),
                        label='normal',
                        confidence=0.9,
                        explanation=f"Sample normal value from training data"
                    ))
        
        return examples
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomaly in a single value using enhanced detection.
        
        Args:
            value: The value to check for anomaly
            context: Additional context information
            
        Returns:
            AnomalyError if an anomaly is detected, None otherwise
        """
        if not self.is_initialized:
            return None
        
        try:
            # Use AnomalyLLM-enhanced detection if available
            if self.anomalyllm_integration:
                # Prepare dynamic context if available
                dynamic_context = self._prepare_dynamic_context(context)
                
                # Detect anomalies with AnomalyLLM
                results = self.anomalyllm_integration.detect_anomalies(
                    [str(value)],
                    threshold=self.threshold,
                    context=[dynamic_context] if dynamic_context else None
                )
                
                if results and results[0]['is_anomaly']:
                    result = results[0]
                    return AnomalyError(
                        anomaly_type="ENHANCED_ML_ANOMALY",
                        probability=1.0 - result['probability_of_correctness'],
                        details={
                            "field": self.field_name,
                            "value": str(value),
                            "explanation": result.get('explanation', 'Enhanced ML anomaly detected'),
                            "global_similarity": result.get('global_similarity', 0.0),
                            "local_similarity": result.get('local_similarity', 0.0),
                            "context_similarities": result.get('context_similarities', []),
                            "detection_method": "anomalyllm_enhanced"
                        }
                    )
            else:
                # Fallback to standard detection
                from anomaly_detectors.ml_based.check_anomalies import check_anomalies
                results = check_anomalies(
                    self.base_model, 
                    [str(value)], 
                    self.threshold,
                    self.reference_centroid
                )
                
                if results and results[0]['is_anomaly']:
                    result = results[0]
                    return AnomalyError(
                        anomaly_type="ML_ANOMALY",
                        probability=1.0 - result['probability_of_correctness'],
                        details={
                            "field": self.field_name,
                            "value": str(value),
                            "explanation": f"Standard ML anomaly detected with similarity {result['probability_of_correctness']:.3f}",
                            "detection_method": "standard_ml"
                        }
                    )
            
            return None
            
        except Exception as e:
            print(f"Error in enhanced anomaly detection: {e}")
            return None
    
    def _prepare_dynamic_context(self, context: Optional[Dict[str, Any]]) -> Optional[DynamicContext]:
        """Prepare dynamic context from the provided context dictionary."""
        if not context:
            return None
        
        dynamic_context = DynamicContext()
        
        # Extract temporal information
        if 'timestamp' in context:
            try:
                dynamic_context.timestamp = pd.to_datetime(context['timestamp'])
            except:
                pass
        
        if 'sequence_position' in context:
            dynamic_context.sequence_position = context['sequence_position']
        
        # Extract metadata
        metadata = {}
        for key, value in context.items():
            if key not in ['timestamp', 'sequence_position']:
                metadata[key] = value
        
        if metadata:
            dynamic_context.metadata = metadata
        
        return dynamic_context
    
    def add_few_shot_example(self, value: str, label: str, confidence: float = 0.9, 
                           explanation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Add a few-shot example for in-context learning.
        
        Args:
            value: The example value
            label: 'normal' or 'anomaly'
            confidence: Confidence in this example
            explanation: Optional explanation
            context: Optional context information
        """
        example = FewShotExample(
            value=value,
            label=label,
            confidence=confidence,
            explanation=explanation,
            context=context
        )
        
        self.few_shot_examples.append(example)
        
        # Update the AnomalyLLM integration if already initialized
        if self.anomalyllm_integration:
            self.anomalyllm_integration.in_context_detector.add_few_shot_examples([example])
    
    def get_detector_args(self) -> Dict[str, Any]:
        """Return arguments needed to recreate this detector instance."""
        return {
            'field_name': self.field_name,
            'threshold': self.threshold,
            'results_dir': self.results_dir,
            'use_gpu': self.use_gpu,
            'enable_anomalyllm': self.enable_anomalyllm,
            'few_shot_examples': self.few_shot_examples,
            'temporal_column': self.temporal_column,
            'context_columns': self.context_columns
        }
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about the AnomalyLLM enhancements."""
        info = {
            'anomalyllm_enabled': self.enable_anomalyllm,
            'few_shot_examples_count': len(self.few_shot_examples),
            'temporal_column': self.temporal_column,
            'context_columns': self.context_columns,
            'is_initialized': self.is_initialized
        }
        
        if self.anomalyllm_integration:
            info.update({
                'dynamic_encoding_enabled': self.anomalyllm_integration.enable_dynamic_encoding,
                'prototype_reprogramming_enabled': self.anomalyllm_integration.enable_prototype_reprogramming,
                'in_context_learning_enabled': self.anomalyllm_integration.enable_in_context_learning
            })
        
        return info


class EnhancedMLAnomalyDetectorFactory:
    """
    Factory for creating enhanced ML anomaly detectors with AnomalyLLM integration.
    """
    
    def __init__(self, 
                 results_dir: str = None,
                 threshold: float = 0.6,
                 enable_anomalyllm: bool = True):
        self.results_dir = results_dir
        self.threshold = threshold
        self.enable_anomalyllm = enable_anomalyllm
    
    def get_detector(self, field_name: str, 
                    few_shot_examples: Optional[List[FewShotExample]] = None,
                    temporal_column: Optional[str] = None,
                    context_columns: Optional[List[str]] = None) -> EnhancedMLAnomalyDetector:
        """
        Create an enhanced ML detector for the specified field.
        
        Args:
            field_name: Name of the field to create detector for
            few_shot_examples: Optional few-shot examples
            temporal_column: Optional temporal column
            context_columns: Optional context columns
            
        Returns:
            Enhanced ML anomaly detector
        """
        return EnhancedMLAnomalyDetector(
            field_name=field_name,
            threshold=self.threshold,
            results_dir=self.results_dir,
            use_gpu=True,
            enable_anomalyllm=self.enable_anomalyllm,
            few_shot_examples=few_shot_examples,
            temporal_column=temporal_column,
            context_columns=context_columns
        )
    
    def list_available_fields(self) -> List[str]:
        """List available fields that have trained models."""
        if not self.results_dir or not os.path.exists(self.results_dir):
            return []
        
        available_fields = []
        field_to_column = get_field_to_column_map()
        
        for field_name in field_to_column.keys():
            model_dir = os.path.join(self.results_dir, f'results_{field_name.replace(" ", "_").lower()}')
            if os.path.isdir(model_dir):
                available_fields.append(field_name)
        
        return available_fields


def create_enhanced_ml_detector_for_field(field_name: str, 
                                        results_dir: str = None,
                                        threshold: float = 0.6,
                                        enable_anomalyllm: bool = True,
                                        few_shot_examples: Optional[List[FewShotExample]] = None,
                                        temporal_column: Optional[str] = None,
                                        context_columns: Optional[List[str]] = None) -> EnhancedMLAnomalyDetector:
    """
    Convenience function to create an enhanced ML detector for a field.
    
    Args:
        field_name: Name of the field
        results_dir: Directory containing trained models
        threshold: Anomaly detection threshold
        enable_anomalyllm: Whether to enable AnomalyLLM features
        few_shot_examples: Optional few-shot examples
        temporal_column: Optional temporal column
        context_columns: Optional context columns
        
    Returns:
        Enhanced ML anomaly detector
    """
    return EnhancedMLAnomalyDetector(
        field_name=field_name,
        threshold=threshold,
        results_dir=results_dir,
        use_gpu=True,
        enable_anomalyllm=enable_anomalyllm,
        few_shot_examples=few_shot_examples,
        temporal_column=temporal_column,
        context_columns=context_columns
    )


def create_enhanced_ml_detector_for_column(column_name: str,
                                         results_dir: str = None,
                                         threshold: float = 0.6,
                                         enable_anomalyllm: bool = True,
                                         few_shot_examples: Optional[List[FewShotExample]] = None,
                                         temporal_column: Optional[str] = None,
                                         context_columns: Optional[List[str]] = None) -> Optional[EnhancedMLAnomalyDetector]:
    """
    Create an enhanced ML detector for a column by finding the corresponding field.
    
    Args:
        column_name: Name of the column
        results_dir: Directory containing trained models
        threshold: Anomaly detection threshold
        enable_anomalyllm: Whether to enable AnomalyLLM features
        few_shot_examples: Optional few-shot examples
        temporal_column: Optional temporal column
        context_columns: Optional context columns
        
    Returns:
        Enhanced ML anomaly detector or None if no matching field found
    """
    field_to_column = get_field_to_column_map()
    
    # Find field name for this column
    field_name = None
    for field, col in field_to_column.items():
        if col == column_name:
            field_name = field
            break
    
    if field_name is None:
        print(f"No field mapping found for column '{column_name}'")
        return None
    
    return create_enhanced_ml_detector_for_field(
        field_name=field_name,
        results_dir=results_dir,
        threshold=threshold,
        enable_anomalyllm=enable_anomalyllm,
        few_shot_examples=few_shot_examples,
        temporal_column=temporal_column,
        context_columns=context_columns
    ) 