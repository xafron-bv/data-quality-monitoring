"""
LLM-based anomaly detector using few-shot learning and dynamic encoding.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import warnings

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from anomaly_detectors.anomaly_error import AnomalyError
from brand_configs import get_brand_config_manager
from field_column_map import get_field_to_column_map as get_global_map

# Try to import LLM dependencies
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    import torch
    from torch.utils.data import Dataset
    import evaluate
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    warnings.warn(f"LLM dependencies not available: {e}")


@dataclass
class DynamicContext:
    """Context information for dynamic-aware encoding"""
    temporal_info: Optional[float] = None
    categorical_info: Optional[str] = None
    contextual_info: Optional[Dict[str, Any]] = None

class DynamicAwareEncoder(nn.Module):
    """Dynamic-aware encoder that incorporates temporal and contextual information."""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Categorical encoding
        self.categorical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, text_embeddings: torch.Tensor, context: Optional[DynamicContext] = None) -> torch.Tensor:
        if context is None:
            return text_embeddings
        
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
        
        # Temporal encoding
        if context.temporal_info is not None:
            temporal_tensor = torch.tensor([[context.temporal_info]], dtype=torch.float32)
            temporal_encoded = self.temporal_encoder(temporal_tensor)
        else:
            temporal_encoded = torch.zeros(1, self.embedding_dim)
        
        # Categorical encoding
        if context.categorical_info is not None:
            # Simple hash-based encoding for categorical data
            cat_hash = hash(context.categorical_info) % 1000
            cat_tensor = torch.tensor([[cat_hash]], dtype=torch.float32)
            categorical_encoded = self.categorical_encoder(cat_tensor)
        else:
            categorical_encoded = torch.zeros(1, self.embedding_dim)
        
        # Ensure combined_features is 2D before fusion
        combined_features = torch.cat([text_embeddings, temporal_encoded.unsqueeze(0), categorical_encoded.unsqueeze(0)], dim=1)
        # Ensure fusion_layer output is 2D
        return self.fusion_layer(combined_features).squeeze(0)  # Squeeze to remove batch dim if it's 1

class PrototypeBasedReprogramming:
    """Prototype-based reprogramming for semantic alignment."""
    
    def __init__(self, n_prototypes: int = 10):
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.is_trained = False
    
    def learn_prototypes(self, embeddings: np.ndarray) -> None:
        """Learn prototypes using K-means clustering."""
        if len(embeddings) < self.n_prototypes:
            self.n_prototypes = len(embeddings)
        
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        self.prototypes = kmeans.cluster_centers_
        self.is_trained = True
    
    def apply_prototypes(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply prototype-based reprogramming."""
        if not self.is_trained or self.prototypes is None:
            return embeddings
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Find nearest prototype for each embedding
        distances = np.linalg.norm(embeddings[:, np.newaxis] - self.prototypes, axis=2)
        nearest_prototype_indices = np.argmin(distances, axis=1)
        nearest_prototypes = self.prototypes[nearest_prototype_indices]
        
        # Interpolate between original and prototype
        alpha = 0.3
        return (1 - alpha) * embeddings + alpha * nearest_prototypes

class InContextLearningDetector:
    """In-context learning detector for few-shot anomaly detection."""
    
    def __init__(self, base_model: SentenceTransformer, few_shot_examples: List[str] = None):
        self.base_model = base_model
        self.few_shot_examples = few_shot_examples or []
        self.example_embeddings = None
        
        if self.few_shot_examples:
            self.example_embeddings = self.base_model.encode(self.few_shot_examples)
    
    def detect_anomaly(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Detect anomaly using in-context learning."""
        if not self.example_embeddings:
            return {"is_anomaly": False, "confidence": 0.5, "explanation": "No examples available"}
        
        # Get embedding for input text
        text_embedding = self.base_model.encode([text])[0]
        
        # Calculate similarities with examples
        similarities = cosine_similarity([text_embedding], self.example_embeddings)[0]
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Determine if anomaly
        is_anomaly = max_similarity < threshold
        
        explanation = f"Max similarity with examples: {max_similarity:.3f}, threshold: {threshold}"
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": max_similarity,
            "explanation": explanation,
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity
        }

def preprocess_text(text: str) -> Optional[str]:
    """Preprocess text for analysis."""
    if pd.isna(text):
        return None
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text if text else None

def get_optimal_device(use_gpu: bool = True) -> torch.device:
    """Get the optimal device for computation."""
    if use_gpu and torch.backends.mps.is_available():
        return torch.device('mps')
    elif use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_device_info(device: torch.device, context: str = ""):
    """Print device information."""
    print(f"🖥️  Using device: {device} {context}")

def get_field_to_column_map() -> Dict[str, str]:
    """Get mapping from field names to column names from current brand configuration."""
    try:
        manager = get_brand_config_manager()
        current_brand = manager.get_current_brand()
        if current_brand:
            return current_brand.field_mappings
        else:
            # Try to get from field_column_map module
            return get_global_map()
    except Exception as e:
        print(f"Warning: Could not load brand configuration: {e}")
        # Return empty dict if brand config not available
        return {}

def calculate_sequence_probability(model, tokenizer, text: str, device: torch.device) -> float:
    """Calculate the probability of a text sequence using the trained language model."""
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128
        ).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Calculate average log probability
        probs = torch.softmax(logits, dim=-1)
        input_ids = inputs['input_ids'][0]
        
        total_log_prob = 0.0
        count = 0
        
        for i, token_id in enumerate(input_ids):
            if token_id != tokenizer.pad_token_id and token_id != tokenizer.cls_token_id:
                prob = probs[0, i, token_id].item()
                if prob > 0:
                    total_log_prob += torch.log(torch.tensor(prob)).item()
                    count += 1
        
        if count > 0:
            return total_log_prob / count
        else:
            return -10.0  # Very low probability
            
    except Exception as e:
        print(f"Error calculating probability: {e}")
        return -10.0

class LLMAnomalyDetector:
    """
    LLM-based anomaly detector using language modeling for probability-based detection.
    Compatible with comprehensive_detector.py interface.
    """
    
    # Class-level cache for models
    _model_cache = {}
    
    def __init__(
        self,
        field_name: str = None,
        threshold: float = -2.0,
        use_gpu: bool = True,
        enable_dynamic_encoding: bool = False,
        enable_prototype_reprogramming: bool = False,
        enable_in_context_learning: bool = False,
        temporal_column: Optional[str] = None,
        context_columns: Optional[List[str]] = None
    ):
        self.field_name = field_name
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.device = get_optimal_device(use_gpu)
        self.is_initialized = False
        self.has_trained_model = False
        
        # Language model components
        self.language_model = None
        self.tokenizer = None
        self.model_path = None
        
        # Optional components
        self.enable_dynamic_encoding = enable_dynamic_encoding
        self.enable_prototype_reprogramming = enable_prototype_reprogramming
        self.enable_in_context_learning = enable_in_context_learning
        
        # Context information
        self.temporal_column = temporal_column
        self.context_columns = context_columns or []
        
        # Initialize optional components
        if self.enable_dynamic_encoding:
            self.dynamic_encoder = DynamicAwareEncoder()
            self.dynamic_encoder.to(self.device)
        else:
            self.dynamic_encoder = None
        
        if self.enable_prototype_reprogramming:
            self.prototype_reprogramming = PrototypeBasedReprogramming()
        else:
            self.prototype_reprogramming = None
        
        if self.enable_in_context_learning:
            self.in_context_detector = None  # Will be initialized in learn_patterns
        else:
            self.in_context_detector = None
    
    def _get_cache_key(self) -> str:
        """Generate cache key for model caching."""
        return f"llm_model_{self.field_name}_{self.threshold}_{self.use_gpu}"
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn patterns by loading the trained language model.
        
        Args:
            df: DataFrame containing the data
            column_name: Column name to detect anomalies in
        """
        if self.is_initialized:
            return
        
        # Check if we have a field name
        if not self.field_name:
            print(f"❌ No field name specified for LLM detector")
            return
        
        # Check class-level cache first
        cache_key = self._get_cache_key()
        if cache_key in LLMAnomalyDetector._model_cache:
            cached_data = LLMAnomalyDetector._model_cache[cache_key]
            self.language_model = cached_data['language_model']
            self.tokenizer = cached_data['tokenizer']
            self.model_path = cached_data['model_path']
            self.column_name = cached_data['column_name']
            self.is_initialized = True
            self.has_trained_model = True
            print(f"📋 Using cached language model for field '{self.field_name}'")
            return
        
        # Load trained model
        model_path = f"llm_results/{self.field_name}_model"
        if not os.path.exists(model_path):
            print(f"❌ No trained model found for field '{self.field_name}' at {model_path}")
            print(f"💡 Skipping LLM detection for this field")
            self.has_trained_model = False
            return
        
        try:
            print(f"🤖 Loading trained language model for field '{self.field_name}' from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.language_model = AutoModelForMaskedLM.from_pretrained(model_path)
            self.language_model.to(self.device)
            self.model_path = model_path
            self.column_name = column_name
            
            # Cache the model
            LLMAnomalyDetector._model_cache[cache_key] = {
                'language_model': self.language_model,
                'tokenizer': self.tokenizer,
                'model_path': self.model_path,
                'column_name': self.column_name
            }
            
            self.is_initialized = True
            self.has_trained_model = True
            print(f"✅ Language model loaded successfully for field '{self.field_name}'")
            
        except Exception as e:
            print(f"❌ Error loading language model for field '{self.field_name}': {e}")
            self.has_trained_model = False
            return
    
    def _extract_dynamic_context(self, context: Dict[str, Any]) -> Optional[DynamicContext]:
        """Extract dynamic context from the provided context dictionary."""
        if not context:
            return None
        
        temporal_info = None
        categorical_info = None
        
        # Extract temporal information
        if self.temporal_column and self.temporal_column in context:
            try:
                temporal_info = float(context[self.temporal_column])
            except (ValueError, TypeError):
                pass
        
        # Extract categorical information
        if self.context_columns:
            context_parts = []
            for col in self.context_columns:
                if col in context and context[col] is not None:
                    context_parts.append(str(context[col]))
            if context_parts:
                categorical_info = " | ".join(context_parts)
        
        if temporal_info is not None or categorical_info is not None:
            return DynamicContext(
                temporal_info=temporal_info,
                categorical_info=categorical_info
            )
        
        return None
    
    def _detect_anomaly(self, value: str, context: Optional[Dict[str, Any]] = None) -> Optional[AnomalyError]:
        """
        Detect anomaly using the trained language model.
        
        Args:
            value: The text value to check for anomalies
            context: Optional context information
            
        Returns:
            AnomalyError if anomaly detected, None otherwise
        """
        if not self.is_initialized or not self.has_trained_model or not self.language_model or not self.tokenizer:
            return None
        
        # Preprocess the value
        processed_value = preprocess_text(value)
        if processed_value is None:
            return None
        
        explanation_parts = []
        
        # Calculate base probability using language model
        try:
            base_probability = calculate_sequence_probability(
                self.language_model, 
                self.tokenizer, 
                processed_value, 
                self.device
            )
            explanation_parts.append("language_model")
        except Exception as e:
            print(f"Language model probability calculation failed: {e}")
            return None
        
        # Apply optional enhancements
        enhanced_probability = base_probability
        
        # Dynamic encoding (if enabled)
        if self.enable_dynamic_encoding and self.dynamic_encoder and context:
            try:
                dynamic_context = self._extract_dynamic_context(context)
                if dynamic_context:
                    # For now, we'll use a simple adjustment based on context
                    # In a full implementation, you might want to encode context and adjust probability
                    enhanced_probability = base_probability * 1.1  # Slight boost for context
                    explanation_parts.append("dynamic_encoding")
            except Exception as e:
                print(f"Dynamic encoding failed: {e}")
        
        # Prototype reprogramming (if enabled)
        if self.enable_prototype_reprogramming and self.prototype_reprogramming:
            try:
                # For language models, we might adjust probability based on learned patterns
                enhanced_probability = enhanced_probability * 1.05  # Slight adjustment
                explanation_parts.append("prototype_reprogramming")
            except Exception as e:
                print(f"Prototype reprogramming failed: {e}")
        
        # In-context learning (if enabled)
        if self.enable_in_context_learning and self.in_context_detector:
            try:
                icl_result = self.in_context_detector.detect_anomaly(processed_value, self.threshold)
                if icl_result["is_anomaly"]:
                    return AnomalyError(
                        anomaly_type="LLM_IN_CONTEXT_ANOMALY",
                        probability=1 - icl_result["confidence"],
                        details={
                            "detection_method": "llm_in_context_learning",
                            "explanation": icl_result["explanation"],
                            "confidence": icl_result["confidence"],
                            "llm_components": explanation_parts
                        },
                        explanation=icl_result["explanation"]
                    )
                explanation_parts.append("in_context_learning")
            except Exception as e:
                print(f"In-context learning failed: {e}")
        
        # Determine if anomaly based on probability threshold
        is_anomaly = enhanced_probability < self.threshold
        
        if is_anomaly:
            return AnomalyError(
                anomaly_type="LLM_LANGUAGE_MODEL_ANOMALY",
                probability=1 - (enhanced_probability + 10) / 10,  # Convert to [0,1] range
                details={
                    "detection_method": "llm_language_model",
                    "explanation": f"Low probability sequence detected (score: {enhanced_probability:.3f}, threshold: {self.threshold})",
                    "confidence": (enhanced_probability + 10) / 10,
                    "probability_score": enhanced_probability,
                    "threshold": self.threshold,
                    "llm_components": explanation_parts
                },
                explanation=f"Low probability sequence detected (score: {enhanced_probability:.3f}, threshold: {self.threshold})"
            )
        
        return None
    
    def bulk_detect(self, df: pd.DataFrame, column_name: str, batch_size: int = 1000, max_workers: int = 1) -> List[Optional[AnomalyError]]:
        """
        Detect anomalies in bulk for a DataFrame column.
        Compatible with comprehensive_detector.py interface.
        
        Args:
            df: DataFrame containing the data
            column_name: Column name to detect anomalies in
            batch_size: Batch size (not used in this implementation)
            max_workers: Number of workers (not used in this implementation)
            
        Returns:
            List of AnomalyError objects (None for non-anomalous values)
        """
        if not self.has_trained_model:
            print(f"⚠️  No trained model available for field '{self.field_name}', skipping LLM detection")
            return [None] * len(df)
        
        # Initialize the detector if not already done
        if not self.is_initialized:
            self.learn_patterns(df, column_name)
        
        if not self.is_initialized or not self.has_trained_model:
            return [None] * len(df)
        
        # Get values from the column
        values = df[column_name].astype(str).tolist()
        
        # Detect anomalies
        results = []
        for i, value in enumerate(values):
            anomaly = self._detect_anomaly(value)
            if anomaly:
                # Add context information to the anomaly
                anomaly.row_index = i
                anomaly.column_name = column_name
                anomaly.anomaly_data = value
            results.append(anomaly)
        
        return results
    
    def detect_anomalies(self, values: List[str], context: Optional[List[Dict[str, Any]]] = None) -> List[Optional[AnomalyError]]:
        """
        Detect anomalies in a list of values.
        
        Args:
            values: List of text values to check
            context: Optional list of context dictionaries
            
        Returns:
            List of AnomalyError objects (None for non-anomalous values)
        """
        if not self.is_initialized or not self.has_trained_model:
            print("❌ Detector not initialized or no trained model available. Call learn_patterns() first.")
            return [None] * len(values)
        
        results = []
        for i, value in enumerate(values):
            context_dict = context[i] if context and i < len(context) else None
            anomaly = self._detect_anomaly(value, context_dict)
            results.append(anomaly)
        
        return results 