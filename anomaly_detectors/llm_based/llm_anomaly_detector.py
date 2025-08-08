"""
LLM-based anomaly detector using few-shot learning and dynamic encoding.
"""

import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import threading

import evaluate
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from anomaly_detectors.anomaly_error import AnomalyError
from common.brand_config import load_brand_config
from common.field_column_map import get_field_to_column_map as get_global_map


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

def get_field_to_column_map(brand_name: str = "esqualo") -> Dict[str, str]:
    """Get mapping from field names to column names for a brand."""
    config = load_brand_config(brand_name)
    return config.field_mappings

def calculate_sequence_probability(model, tokenizer, text: str, device: torch.device) -> float:
    """Calculate the anomaly score for a text sequence using the trained language model.
    Returns a positive score where higher values indicate more anomalous text."""
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

        # Calculate average negative log probability (higher = more anomalous)
        probs = torch.softmax(logits, dim=-1)
        input_ids = inputs['input_ids'][0]

        total_neg_log_prob = 0.0
        count = 0

        for i, token_id in enumerate(input_ids):
            if token_id != tokenizer.pad_token_id and token_id != tokenizer.cls_token_id:
                prob = probs[0, i, token_id].item()
                if prob > 0:
                    # Use negative log probability - higher values = more anomalous
                    total_neg_log_prob += -torch.log(torch.tensor(prob)).item()
                    count += 1

        if count > 0:
            # Return average negative log probability (positive value)
            return total_neg_log_prob / count
        else:
            return 10.0  # High anomaly score

    except Exception as e:
        print(f"Error calculating probability: {e}")
        return 10.0  # High anomaly score on error

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
        context_columns: Optional[List[str]] = None,
        variation: Optional[str] = None
    ):
        self.field_name = field_name
        # Threshold is now positive - higher values are more anomalous
        self.threshold = abs(threshold)
        self.use_gpu = use_gpu
        self.device = get_optimal_device(use_gpu)
        self.is_initialized = False
        self.has_trained_model = False

        # Language model components
        self.language_model = None
        self.tokenizer = None
        self.model_path = None
        self.variation = variation

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
        return f"llm_model_{self.field_name}_{self.variation}_{self.threshold}_{self.use_gpu}"

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

        if not self.variation:
            print(f"❌ Variation is required for LLM detector (field '{self.field_name}')")
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
            print(f"📋 Using cached language model for field '{self.field_name}' (variation '{self.variation}')")
            return

        # Load trained model from variation-specific path
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "llm", f"{self.field_name}_model", self.variation)
        if not os.path.exists(model_path):
            print(f"❌ No trained model found for field '{self.field_name}' variation '{self.variation}' at {model_path}")
            print(f"💡 Skipping LLM detection for this field")
            self.has_trained_model = False
            return

        try:
            print(f"🤖 Loading trained language model for field '{self.field_name}' variation '{self.variation}' from {model_path}")
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
            print(f"✅ Language model loaded successfully for field '{self.field_name}' (variation '{self.variation}')")

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

        # Determine if anomaly based on threshold (higher score = more anomalous)
        is_anomaly = enhanced_probability > self.threshold

        if is_anomaly:
            # Normalize score to [0,1] range for probability
            # Assuming typical scores range from 0 to 10
            normalized_prob = min(enhanced_probability / 10.0, 1.0)
            
            return AnomalyError(
                anomaly_type="LLM_LANGUAGE_MODEL_ANOMALY",
                probability=normalized_prob,
                details={
                    "detection_method": "llm_language_model",
                    "explanation": f"High anomaly score detected (score: {enhanced_probability:.3f}, threshold: {self.threshold})",
                    "confidence": 1.0 - normalized_prob,
                    "anomaly_score": enhanced_probability,
                    "threshold": self.threshold,
                    "llm_components": explanation_parts
                },
                explanation=f"High anomaly score detected (score: {enhanced_probability:.3f}, threshold: {self.threshold})"
            )

        return None

    def bulk_detect(self, df: pd.DataFrame, column_name: str, batch_size: int = 100, max_workers: int = 4) -> List[Optional[AnomalyError]]:
        """
        Detect anomalies in bulk for a DataFrame column using parallel processing.
        
        Args:
            df: DataFrame containing the data
            column_name: Column name to detect anomalies in
            batch_size: Number of texts to process in each batch
            max_workers: Number of concurrent workers for parallel processing
            
        Returns:
            List of AnomalyError objects (None for non-anomalous values)
        """
        # Initialize the detector if not already done
        if not self.is_initialized:
            self.learn_patterns(df, column_name)
            
        if not self.is_initialized or not self.has_trained_model:
            print(f"⚠️  No trained model available for field '{self.field_name}', skipping LLM detection")
            return [None] * len(df)
            
        # Get values from the column
        values = df[column_name].astype(str).tolist()
        total_rows = len(values)
        
        print(f"[LLM Bulk Detect] Processing {total_rows} rows with batch_size={batch_size}, max_workers={max_workers}")
        
        # If using single worker or small dataset, use batch processing
        if batch_size is None:
            batch_size = 100  # Default batch size
        
        # For now, always use single-threaded processing to avoid parallel issues
        return self._batch_detect(values, column_name, batch_size)
        
        # For multiple workers, split work across processes
        import concurrent.futures
        from functools import partial
        
        # Calculate chunks for each worker
        chunk_size = max(batch_size, (total_rows + max_workers - 1) // max_workers)
        chunks = []
        
        for i in range(0, total_rows, chunk_size):
            chunk_values = values[i:i + chunk_size]
            chunk_indices = list(range(i, min(i + chunk_size, total_rows)))
            chunks.append((chunk_values, chunk_indices))
        
        print(f"[LLM Bulk Detect] Split into {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        results = [None] * total_rows
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk_values, chunk_indices, column_name, batch_size): chunk_idx
                for chunk_idx, (chunk_values, chunk_indices) in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results, chunk_indices = future.result()
                    # Place results in correct positions
                    for idx, result in zip(chunk_indices, chunk_results):
                        results[idx] = result
                except Exception as e:
                    print(f"[LLM Bulk Detect] Chunk {chunk_idx} failed: {e}")
                    # Fill failed chunk with None
                    _, chunk_indices = chunks[chunk_idx]
                    for idx in chunk_indices:
                        results[idx] = None
        
        print(f"[LLM Bulk Detect] Completed processing {total_rows} rows")
        return results
    
    def _create_worker_model(self):
        """Create a copy of the model for a worker thread."""
        import copy
        import os
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        
        # Create a new detector instance with the same configuration
        worker_detector = LLMAnomalyDetector(
            field_name=self.field_name,
            threshold=self.threshold,
            use_gpu=False,  # Use CPU for worker threads to avoid GPU memory issues
            enable_dynamic_encoding=self.enable_dynamic_encoding,
            enable_prototype_reprogramming=self.enable_prototype_reprogramming,
            enable_in_context_learning=self.enable_in_context_learning,
            temporal_column=self.temporal_column,
            context_columns=self.context_columns
        )
        
        # Copy the model state
        if self.model_path and os.path.exists(self.model_path):
            # Load model directly to CPU to avoid meta tensor issues
            worker_detector.language_model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                device_map='cpu',
                torch_dtype=torch.float32
            )
            worker_detector.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            worker_detector.language_model.eval()
            worker_detector.is_initialized = True
            worker_detector.has_trained_model = True
            worker_detector.model_path = self.model_path
            worker_detector.column_name = self.column_name
            worker_detector.device = 'cpu'
            
            # Copy optional components if needed
            if hasattr(self, 'dynamic_encoder') and self.dynamic_encoder:
                worker_detector.dynamic_encoder = copy.deepcopy(self.dynamic_encoder)
            if hasattr(self, 'prototype_reprogrammer') and self.prototype_reprogrammer:
                worker_detector.prototype_reprogrammer = copy.deepcopy(self.prototype_reprogrammer)
            if hasattr(self, 'in_context_learner') and self.in_context_learner:
                worker_detector.in_context_learner = copy.deepcopy(self.in_context_learner)
        
        return worker_detector
    
    def _process_chunk(self, chunk_values: List[str], chunk_indices: List[int], column_name: str, batch_size: int) -> Tuple[List[Optional[AnomalyError]], List[int]]:
        """Process a chunk of values using batch detection."""
        # Use the main model with thread safety
        chunk_results = self._batch_detect(chunk_values, column_name, batch_size)
        
        # Add row indices to anomalies
        for i, (result, row_idx) in enumerate(zip(chunk_results, chunk_indices)):
            if result:
                result.row_index = row_idx
        
        return chunk_results, chunk_indices
    
    def _batch_detect(self, values: List[str], column_name: str, batch_size: int) -> List[Optional[AnomalyError]]:
        """
        Detect anomalies using batch processing for efficiency.
        
        Args:
            values: List of text values to check
            column_name: Column name for context
            batch_size: Number of texts to process together
            
        Returns:
            List of AnomalyError objects (None for non-anomalous values)
        """
        results = []
        total_values = len(values)
        
        # Process in batches
        for batch_start in range(0, total_values, batch_size):
            batch_end = min(batch_start + batch_size, total_values)
            batch_values = values[batch_start:batch_end]
            
            # Preprocess batch
            processed_batch = [preprocess_text(v) for v in batch_values]
            valid_indices = [i for i, v in enumerate(processed_batch) if v is not None]
            
            if not valid_indices:
                # All values in batch are invalid
                results.extend([None] * len(batch_values))
                continue
            
            # Get only valid values for processing
            valid_values = [processed_batch[i] for i in valid_indices]
            
            try:
                # Batch tokenization
                inputs = self.tokenizer(
                    valid_values,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(self.device)
                
                # Batch inference
                with torch.no_grad():
                    outputs = self.language_model(**inputs)
                    logits = outputs.logits
                
                # Calculate probabilities for each text in batch
                batch_probabilities = []
                for i in range(len(valid_values)):
                    # Calculate average log probability for this text
                    probs = torch.softmax(logits[i], dim=-1)
                    input_ids = inputs['input_ids'][i]
                    
                    total_neg_log_prob = 0.0
                    count = 0
                    
                    for j, token_id in enumerate(input_ids):
                        if token_id != self.tokenizer.pad_token_id and token_id != self.tokenizer.cls_token_id:
                            prob = probs[j, token_id].item()
                            if prob > 0:
                                # Use negative log probability - higher values = more anomalous
                                total_neg_log_prob += -torch.log(torch.tensor(prob)).item()
                                count += 1
                    
                    if count > 0:
                        avg_anomaly_score = total_neg_log_prob / count
                    else:
                        avg_anomaly_score = 10.0  # High anomaly score
                        
                    batch_probabilities.append(avg_anomaly_score)
                
                # Create results for this batch
                batch_results = [None] * len(batch_values)
                
                for i, valid_idx in enumerate(valid_indices):
                    anomaly_score = batch_probabilities[i]
                    
                    # Check if anomaly (higher score = more anomalous)
                    if anomaly_score > self.threshold:
                        # Normalize score to [0,1] range
                        normalized_prob = min(anomaly_score / 10.0, 1.0)
                        
                        batch_results[valid_idx] = AnomalyError(
                            anomaly_type="LLM_LANGUAGE_MODEL_ANOMALY",
                            probability=normalized_prob,
                            column_name=column_name,
                            anomaly_data=batch_values[valid_idx],
                            details={
                                "detection_method": "llm_language_model_batch",
                                "explanation": f"High anomaly score detected (score: {anomaly_score:.3f}, threshold: {self.threshold})",
                                "confidence": 1.0 - normalized_prob,
                                "anomaly_score": anomaly_score,
                                "threshold": self.threshold,
                                "batch_processing": True
                            },
                            explanation=f"High anomaly score detected (score: {anomaly_score:.3f}, threshold: {self.threshold})"
                        )
                
                results.extend(batch_results)
                
            except Exception as e:
                print(f"[LLM Batch Detect] Batch processing failed: {e}")
                # Fall back to individual processing for this batch
                for value in batch_values:
                    anomaly = self._detect_anomaly(value)
                    if anomaly:
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


def create_llm_detector_for_field(
    field_name: str,
    threshold: float = -2.0,
    use_gpu: bool = True,
    enable_dynamic_encoding: bool = False,
    enable_prototype_reprogramming: bool = False,
    enable_in_context_learning: bool = False,
    temporal_column: Optional[str] = None,
    context_columns: Optional[List[str]] = None,
    variation: Optional[str] = None
) -> LLMAnomalyDetector:
    """
    Factory function to create an LLM anomaly detector for a specific field.

    Args:
        field_name: Name of the field to detect anomalies for
        threshold: Probability threshold for anomaly detection
        use_gpu: Whether to use GPU for inference
        enable_dynamic_encoding: Enable dynamic context encoding
        enable_prototype_reprogramming: Enable prototype-based reprogramming
        enable_in_context_learning: Enable in-context learning
        temporal_column: Column name for temporal information
        context_columns: List of column names for context

    Returns:
        LLMAnomalyDetector instance
    """
    return LLMAnomalyDetector(
        field_name=field_name,
        threshold=threshold,
        use_gpu=use_gpu,
        enable_dynamic_encoding=enable_dynamic_encoding,
        enable_prototype_reprogramming=enable_prototype_reprogramming,
        enable_in_context_learning=enable_in_context_learning,
        temporal_column=temporal_column,
        context_columns=context_columns,
        variation=variation
    )
