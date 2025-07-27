"""
AnomalyLLM Integration Module

This module integrates key concepts from AnomalyLLM paper into the existing ML-based
anomaly detection system, focusing on:

1. Few-shot anomaly detection using in-context learning
2. Dynamic-aware encoding for temporal data patterns
3. Prototype-based edge reprogramming
4. Enhanced semantic alignment with LLM knowledge

Based on: "AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detectors.ml_based.model_training import preprocess_text
from anomaly_detectors.ml_based.gpu_utils import get_optimal_device, get_optimal_batch_size


@dataclass
class FewShotExample:
    """Represents a few-shot example for in-context learning."""
    value: str
    label: str  # 'normal' or 'anomaly'
    confidence: float
    explanation: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class DynamicContext:
    """Represents dynamic context information for temporal patterns."""
    timestamp: Optional[datetime] = None
    sequence_position: Optional[int] = None
    temporal_features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class DynamicAwareEncoder(nn.Module):
    """
    Dynamic-aware encoder that incorporates temporal and contextual information
    into the embedding process, inspired by AnomalyLLM's dynamic-aware encoder.
    """
    
    def __init__(self, base_model: SentenceTransformer, 
                 temporal_dim: int = 64,
                 context_dim: int = 32,
                 fusion_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim
        self.fusion_dim = fusion_dim
        
        # Temporal encoding layers
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, temporal_dim)
        )
        
        # Context encoding layers
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # Fusion layer to combine base embeddings with temporal/context info
        base_embedding_dim = self.base_model.get_sentence_embedding_dimension()
        self.fusion_layer = nn.Sequential(
            nn.Linear(base_embedding_dim + temporal_dim + context_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, base_embedding_dim),
            nn.LayerNorm(base_embedding_dim)
        )
        
    def forward(self, texts: List[str], 
                temporal_features: Optional[np.ndarray] = None,
                context_features: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Encode texts with dynamic awareness.
        
        Args:
            texts: List of text strings to encode
            temporal_features: Optional temporal features (e.g., timestamps, sequence positions)
            context_features: Optional context features (e.g., metadata, categorical info)
            
        Returns:
            Dynamic-aware embeddings
        """
        # Get base embeddings
        base_embeddings = self.base_model.encode(texts, convert_to_tensor=True)
        
        batch_size = base_embeddings.shape[0]
        
        # Process temporal features
        if temporal_features is not None:
            temporal_embeddings = self.temporal_encoder(
                torch.tensor(temporal_features, dtype=torch.float32).unsqueeze(-1)
            )
        else:
            temporal_embeddings = torch.zeros(batch_size, self.temporal_dim)
        
        # Process context features
        if context_features is not None:
            context_embeddings = self.context_encoder(
                torch.tensor(context_features, dtype=torch.float32)
            )
        else:
            context_embeddings = torch.zeros(batch_size, self.context_dim)
        
        # Concatenate and fuse
        combined = torch.cat([base_embeddings, temporal_embeddings, context_embeddings], dim=1)
        dynamic_embeddings = self.fusion_layer(combined)
        
        return dynamic_embeddings


class PrototypeBasedReprogramming:
    """
    Prototype-based edge reprogramming inspired by AnomalyLLM's approach
    to align graph structures with LLM knowledge.
    """
    
    def __init__(self, prototype_dim: int = 768, num_prototypes: int = 10):
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        self.prototypes = None
        self.prototype_weights = None
        
    def learn_prototypes(self, embeddings: np.ndarray, labels: Optional[List[str]] = None):
        """
        Learn prototypes from embeddings using clustering.
        
        Args:
            embeddings: Input embeddings
            labels: Optional labels for supervised prototype learning
        """
        if labels is not None:
            # Supervised prototype learning
            unique_labels = list(set(labels))
            self.prototypes = []
            self.prototype_weights = []
            
            for label in unique_labels:
                label_embeddings = embeddings[[i for i, l in enumerate(labels) if l == label]]
                if len(label_embeddings) > 0:
                    # Use mean as prototype for this label
                    prototype = np.mean(label_embeddings, axis=0)
                    weight = len(label_embeddings) / len(embeddings)
                    self.prototypes.append(prototype)
                    self.prototype_weights.append(weight)
        else:
            # Unsupervised prototype learning using K-means
            kmeans = KMeans(n_clusters=min(self.num_prototypes, len(embeddings)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            self.prototypes = kmeans.cluster_centers_
            self.prototype_weights = np.bincount(cluster_labels) / len(cluster_labels)
    
    def reprogram_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reprogram embeddings using learned prototypes.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Reprogrammed embeddings
        """
        if self.prototypes is None:
            return embeddings
        
        # Calculate similarities to prototypes
        similarities = cosine_similarity(embeddings, self.prototypes)
        
        # Weighted combination of original embeddings and prototype influence
        prototype_influence = np.dot(similarities, self.prototypes)
        reprogrammed = 0.7 * embeddings + 0.3 * prototype_influence
        
        return reprogrammed


class InContextLearningDetector:
    """
    In-context learning detector for few-shot anomaly detection,
    inspired by AnomalyLLM's in-context learning framework.
    """
    
    def __init__(self, base_model: SentenceTransformer, 
                 dynamic_encoder: Optional[DynamicAwareEncoder] = None,
                 prototype_reprogramming: Optional[PrototypeBasedReprogramming] = None):
        self.base_model = base_model
        self.dynamic_encoder = dynamic_encoder
        self.prototype_reprogramming = prototype_reprogramming
        self.few_shot_examples: List[FewShotExample] = []
        self.example_embeddings: Optional[np.ndarray] = None
        
    def add_few_shot_examples(self, examples: List[FewShotExample]):
        """
        Add few-shot examples for in-context learning.
        
        Args:
            examples: List of few-shot examples
        """
        self.few_shot_examples.extend(examples)
        self._update_example_embeddings()
    
    def _update_example_embeddings(self):
        """Update embeddings for few-shot examples."""
        if not self.few_shot_examples:
            return
        
        texts = [ex.value for ex in self.few_shot_examples]
        processed_texts = [preprocess_text(text) for text in texts]
        
        if self.dynamic_encoder:
            # Use dynamic-aware encoding
            embeddings = self.dynamic_encoder(processed_texts).detach().numpy()
        else:
            # Use base model encoding
            embeddings = self.base_model.encode(processed_texts, convert_to_numpy=True)
        
        if self.prototype_reprogramming:
            # Apply prototype-based reprogramming
            embeddings = self.prototype_reprogramming.reprogram_embeddings(embeddings)
        
        self.example_embeddings = embeddings
    
    def detect_anomalies_with_context(self, 
                                    values: List[str],
                                    threshold: float = 0.6,
                                    context: Optional[List[DynamicContext]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies using in-context learning with few-shot examples.
        
        Args:
            values: Values to check for anomalies
            threshold: Anomaly detection threshold
            context: Optional dynamic context for each value
            
        Returns:
            List of anomaly detection results
        """
        if not self.few_shot_examples:
            # Fallback to standard detection if no examples provided
            return self._standard_detection(values, threshold)
        
        # Process input values
        processed_values = [preprocess_text(value) for value in values]
        
        # Encode input values
        if self.dynamic_encoder and context:
            # Extract temporal and context features
            temporal_features = []
            context_features = []
            
            for ctx in context:
                if ctx.temporal_features is not None:
                    temporal_features.append(ctx.temporal_features)
                else:
                    temporal_features.append([0.0])  # Default temporal feature
                
                if ctx.metadata:
                    # Convert metadata to context features
                    ctx_feat = self._extract_context_features(ctx.metadata)
                    context_features.append(ctx_feat)
                else:
                    context_features.append(np.zeros(self.dynamic_encoder.context_dim))
            
            temporal_features = np.array(temporal_features)
            context_features = np.array(context_features)
            
            embeddings = self.dynamic_encoder(
                processed_values, 
                temporal_features, 
                context_features
            ).detach().numpy()
        else:
            embeddings = self.base_model.encode(processed_values, convert_to_numpy=True)
        
        if self.prototype_reprogramming:
            embeddings = self.prototype_reprogramming.reprogram_embeddings(embeddings)
        
        # Calculate similarities to few-shot examples
        similarities = cosine_similarity(embeddings, self.example_embeddings)
        
        results = []
        for i, value in enumerate(values):
            # Find most similar examples
            example_similarities = similarities[i]
            top_indices = np.argsort(example_similarities)[::-1][:3]  # Top 3 similar examples
            
            # Weighted voting based on similarities
            normal_score = 0.0
            anomaly_score = 0.0
            
            for idx in top_indices:
                example = self.few_shot_examples[idx]
                similarity = example_similarities[idx]
                
                if example.label == 'normal':
                    normal_score += similarity * example.confidence
                else:
                    anomaly_score += similarity * example.confidence
            
            # Determine final score and prediction
            total_score = normal_score + anomaly_score
            if total_score > 0:
                normal_probability = normal_score / total_score
                is_anomaly = normal_probability < threshold
            else:
                normal_probability = 0.5
                is_anomaly = False
            
            # Generate explanation
            explanation = self._generate_explanation(value, top_indices, example_similarities)
            
            results.append({
                'value': value,
                'is_anomaly': is_anomaly,
                'probability_of_correctness': normal_probability,
                'explanation': explanation,
                'context_similarities': example_similarities[top_indices].tolist()
            })
        
        return results
    
    def _standard_detection(self, values: List[str], threshold: float) -> List[Dict[str, Any]]:
        """Fallback to standard centroid-based detection."""
        processed_values = [preprocess_text(value) for value in values]
        embeddings = self.base_model.encode(processed_values, convert_to_numpy=True)
        
        # Simple outlier detection using embedding statistics
        centroid = np.mean(embeddings, axis=0)
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        
        results = []
        for i, value in enumerate(values):
            similarity = float(similarities[i])
            is_anomaly = similarity < threshold
            
            results.append({
                'value': value,
                'is_anomaly': is_anomaly,
                'probability_of_correctness': similarity,
                'explanation': f"Standard detection with similarity {similarity:.3f}"
            })
        
        return results
    
    def _extract_context_features(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract context features from metadata."""
        # Simple feature extraction - can be enhanced based on specific metadata structure
        features = []
        
        # Extract categorical features
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple hash-based encoding
                features.append(hash(value) % 1000 / 1000.0)
            else:
                features.append(0.0)
        
        # Pad or truncate to required dimension
        target_dim = self.dynamic_encoder.context_dim if self.dynamic_encoder else 32
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return np.array(features)
    
    def _generate_explanation(self, value: str, top_indices: np.ndarray, 
                            similarities: np.ndarray) -> str:
        """Generate explanation based on similar few-shot examples."""
        explanations = []
        
        for idx in top_indices:
            example = self.few_shot_examples[idx]
            similarity = similarities[idx]
            
            if similarity > 0.5:  # Only include reasonably similar examples
                explanation_part = f"Similar to '{example.value}' (similarity: {similarity:.3f})"
                if example.explanation:
                    explanation_part += f" - {example.explanation}"
                explanations.append(explanation_part)
        
        if explanations:
            return "; ".join(explanations[:2])  # Limit to top 2 explanations
        else:
            return f"No similar examples found for '{value}'"


class AnomalyLLMIntegration:
    """
    Main integration class that combines all AnomalyLLM concepts
    into the existing anomaly detection framework.
    """
    
    def __init__(self, 
                 base_model: SentenceTransformer,
                 enable_dynamic_encoding: bool = True,
                 enable_prototype_reprogramming: bool = True,
                 enable_in_context_learning: bool = True):
        self.base_model = base_model
        self.enable_dynamic_encoding = enable_dynamic_encoding
        self.enable_prototype_reprogramming = enable_prototype_reprogramming
        self.enable_in_context_learning = enable_in_context_learning
        
        # Initialize components
        self.dynamic_encoder = None
        self.prototype_reprogramming = None
        self.in_context_detector = None
        
        if enable_dynamic_encoding:
            self.dynamic_encoder = DynamicAwareEncoder(base_model)
        
        if enable_prototype_reprogramming:
            self.prototype_reprogramming = PrototypeBasedReprogramming()
        
        if enable_in_context_learning:
            self.in_context_detector = InContextLearningDetector(
                base_model, self.dynamic_encoder, self.prototype_reprogramming
            )
    
    def train_with_anomalyllm_concepts(self, 
                                     df: pd.DataFrame,
                                     column_name: str,
                                     field_name: str,
                                     few_shot_examples: Optional[List[FewShotExample]] = None,
                                     temporal_column: Optional[str] = None,
                                     context_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the model with AnomalyLLM concepts.
        
        Args:
            df: Training data
            column_name: Target column for anomaly detection
            field_name: Field name for model identification
            few_shot_examples: Optional few-shot examples for in-context learning
            temporal_column: Optional column containing temporal information
            context_columns: Optional columns containing contextual information
            
        Returns:
            Training results and model artifacts
        """
        results = {
            'field_name': field_name,
            'column_name': column_name,
            'training_timestamp': datetime.now().isoformat(),
            'anomalyllm_features': {
                'dynamic_encoding': self.enable_dynamic_encoding,
                'prototype_reprogramming': self.enable_prototype_reprogramming,
                'in_context_learning': self.enable_in_context_learning
            }
        }
        
        # Prepare training data
        texts = df[column_name].dropna().apply(preprocess_text).astype(str).tolist()
        
        # Extract temporal and context features if available
        temporal_features = None
        context_features = None
        
        if temporal_column and temporal_column in df.columns:
            temporal_features = self._extract_temporal_features(df[temporal_column])
        
        if context_columns:
            context_features = self._extract_context_features(df, context_columns)
        
        # Train dynamic encoder if enabled
        if self.dynamic_encoder and (temporal_features is not None or context_features is not None):
            print("Training dynamic-aware encoder...")
            self._train_dynamic_encoder(texts, temporal_features, context_features)
            results['dynamic_encoder_trained'] = True
        
        # Learn prototypes if enabled
        if self.prototype_reprogramming:
            print("Learning prototypes...")
            base_embeddings = self.base_model.encode(texts, convert_to_numpy=True)
            self.prototype_reprogramming.learn_prototypes(base_embeddings)
            results['prototypes_learned'] = True
            results['num_prototypes'] = len(self.prototype_reprogramming.prototypes)
        
        # Setup in-context learning if enabled
        if self.in_context_detector and few_shot_examples:
            print("Setting up in-context learning...")
            self.in_context_detector.add_few_shot_examples(few_shot_examples)
            results['in_context_learning_setup'] = True
            results['num_few_shot_examples'] = len(few_shot_examples)
        
        return results
    
    def detect_anomalies(self, 
                        values: List[str],
                        threshold: float = 0.6,
                        context: Optional[List[DynamicContext]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies using AnomalyLLM-enhanced approach.
        
        Args:
            values: Values to check for anomalies
            threshold: Anomaly detection threshold
            context: Optional dynamic context for each value
            
        Returns:
            List of anomaly detection results
        """
        if self.in_context_detector and self.in_context_detector.few_shot_examples:
            # Use in-context learning if examples are available
            return self.in_context_detector.detect_anomalies_with_context(
                values, threshold, context
            )
        else:
            # Fallback to enhanced standard detection
            return self._enhanced_standard_detection(values, threshold, context)
    
    def _train_dynamic_encoder(self, texts: List[str], 
                             temporal_features: Optional[np.ndarray],
                             context_features: Optional[np.ndarray]):
        """Train the dynamic encoder."""
        # Simple training approach - can be enhanced with more sophisticated training
        if temporal_features is None:
            temporal_features = np.zeros((len(texts), 1))
        
        if context_features is None:
            context_features = np.zeros((len(texts), self.dynamic_encoder.context_dim))
        
        # For now, we'll use the encoder in evaluation mode
        # In a full implementation, you would train this with a specific objective
        self.dynamic_encoder.eval()
    
    def _extract_temporal_features(self, temporal_series: pd.Series) -> np.ndarray:
        """Extract temporal features from temporal column."""
        features = []
        
        for value in temporal_series:
            if pd.isna(value):
                features.append([0.0])
            elif isinstance(value, (int, float)):
                features.append([float(value)])
            else:
                # Try to parse as datetime
                try:
                    dt = pd.to_datetime(value)
                    # Extract various temporal features
                    features.append([
                        dt.hour / 24.0,  # Hour of day (normalized)
                        dt.day / 31.0,   # Day of month (normalized)
                        dt.month / 12.0, # Month (normalized)
                        dt.weekday() / 7.0  # Day of week (normalized)
                    ])
                except:
                    features.append([0.0])
        
        return np.array(features)
    
    def _extract_context_features(self, df: pd.DataFrame, context_columns: List[str]) -> np.ndarray:
        """Extract context features from context columns."""
        features = []
        
        for _, row in df.iterrows():
            row_features = []
            for col in context_columns:
                if col in row:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        row_features.append(float(value))
                    elif isinstance(value, str):
                        row_features.append(hash(value) % 1000 / 1000.0)
                    else:
                        row_features.append(0.0)
                else:
                    row_features.append(0.0)
            
            # Pad to required dimension
            target_dim = self.dynamic_encoder.context_dim if self.dynamic_encoder else 32
            if len(row_features) < target_dim:
                row_features.extend([0.0] * (target_dim - len(row_features)))
            else:
                row_features = row_features[:target_dim]
            
            features.append(row_features)
        
        return np.array(features)
    
    def _enhanced_standard_detection(self, values: List[str], 
                                   threshold: float,
                                   context: Optional[List[DynamicContext]] = None) -> List[Dict[str, Any]]:
        """Enhanced standard detection with AnomalyLLM concepts."""
        processed_values = [preprocess_text(value) for value in values]
        
        # Encode with dynamic awareness if available
        if self.dynamic_encoder and context:
            temporal_features = []
            context_features = []
            
            for ctx in context:
                if ctx.temporal_features is not None:
                    temporal_features.append(ctx.temporal_features)
                else:
                    temporal_features.append([0.0])
                
                if ctx.metadata:
                    ctx_feat = self._extract_context_features_from_metadata(ctx.metadata)
                    context_features.append(ctx_feat)
                else:
                    context_features.append(np.zeros(self.dynamic_encoder.context_dim))
            
            embeddings = self.dynamic_encoder(
                processed_values,
                np.array(temporal_features),
                np.array(context_features)
            ).detach().numpy()
        else:
            embeddings = self.base_model.encode(processed_values, convert_to_numpy=True)
        
        # Apply prototype reprogramming if available
        if self.prototype_reprogramming:
            embeddings = self.prototype_reprogramming.reprogram_embeddings(embeddings)
        
        # Enhanced outlier detection
        centroid = np.mean(embeddings, axis=0)
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        
        # Local neighborhood analysis
        pairwise_sims = cosine_similarity(embeddings)
        local_sims = np.mean(pairwise_sims, axis=1)
        
        # Combine global and local analysis
        final_scores = np.minimum(similarities, local_sims)
        
        results = []
        for i, value in enumerate(values):
            similarity = float(final_scores[i])
            is_anomaly = similarity < threshold
            
            explanation = f"AnomalyLLM-enhanced detection: global similarity {similarities[i]:.3f}, local similarity {local_sims[i]:.3f}"
            
            results.append({
                'value': value,
                'is_anomaly': is_anomaly,
                'probability_of_correctness': similarity,
                'explanation': explanation,
                'global_similarity': float(similarities[i]),
                'local_similarity': float(local_sims[i])
            })
        
        return results
    
    def _extract_context_features_from_metadata(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract context features from metadata dictionary."""
        features = []
        
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(hash(value) % 1000 / 1000.0)
            else:
                features.append(0.0)
        
        target_dim = self.dynamic_encoder.context_dim if self.dynamic_encoder else 32
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return np.array(features)
    
    def save_model(self, model_dir: str):
        """Save the AnomalyLLM-enhanced model."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save base model
        self.base_model.save(model_dir)
        
        # Save AnomalyLLM components
        anomalyllm_config = {
            'enable_dynamic_encoding': self.enable_dynamic_encoding,
            'enable_prototype_reprogramming': self.enable_prototype_reprogramming,
            'enable_in_context_learning': self.enable_in_context_learning,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(model_dir, 'anomalyllm_config.json'), 'w') as f:
            json.dump(anomalyllm_config, f, indent=2)
        
        # Save dynamic encoder if available
        if self.dynamic_encoder:
            torch.save(self.dynamic_encoder.state_dict(), 
                      os.path.join(model_dir, 'dynamic_encoder.pth'))
        
        # Save prototypes if available
        if self.prototype_reprogramming and self.prototype_reprogramming.prototypes is not None:
            np.save(os.path.join(model_dir, 'prototypes.npy'), 
                   self.prototype_reprogramming.prototypes)
            np.save(os.path.join(model_dir, 'prototype_weights.npy'), 
                   self.prototype_reprogramming.prototype_weights)
        
        # Save few-shot examples if available
        if self.in_context_detector and self.in_context_detector.few_shot_examples:
            examples_data = []
            for ex in self.in_context_detector.few_shot_examples:
                examples_data.append({
                    'value': ex.value,
                    'label': ex.label,
                    'confidence': ex.confidence,
                    'explanation': ex.explanation,
                    'context': ex.context
                })
            
            with open(os.path.join(model_dir, 'few_shot_examples.json'), 'w') as f:
                json.dump(examples_data, f, indent=2)
    
    @classmethod
    def load_model(cls, model_dir: str, base_model: SentenceTransformer) -> 'AnomalyLLMIntegration':
        """Load an AnomalyLLM-enhanced model."""
        config_path = os.path.join(model_dir, 'anomalyllm_config.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AnomalyLLM config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create integration instance
        integration = cls(
            base_model,
            enable_dynamic_encoding=config.get('enable_dynamic_encoding', True),
            enable_prototype_reprogramming=config.get('enable_prototype_reprogramming', True),
            enable_in_context_learning=config.get('enable_in_context_learning', True)
        )
        
        # Load dynamic encoder if available
        if integration.dynamic_encoder:
            encoder_path = os.path.join(model_dir, 'dynamic_encoder.pth')
            if os.path.exists(encoder_path):
                integration.dynamic_encoder.load_state_dict(torch.load(encoder_path))
        
        # Load prototypes if available
        if integration.prototype_reprogramming:
            prototypes_path = os.path.join(model_dir, 'prototypes.npy')
            weights_path = os.path.join(model_dir, 'prototype_weights.npy')
            
            if os.path.exists(prototypes_path) and os.path.exists(weights_path):
                integration.prototype_reprogramming.prototypes = np.load(prototypes_path)
                integration.prototype_reprogramming.prototype_weights = np.load(weights_path)
        
        # Load few-shot examples if available
        if integration.in_context_detector:
            examples_path = os.path.join(model_dir, 'few_shot_examples.json')
            if os.path.exists(examples_path):
                with open(examples_path, 'r') as f:
                    examples_data = json.load(f)
                
                examples = []
                for ex_data in examples_data:
                    examples.append(FewShotExample(
                        value=ex_data['value'],
                        label=ex_data['label'],
                        confidence=ex_data['confidence'],
                        explanation=ex_data.get('explanation'),
                        context=ex_data.get('context')
                    ))
                
                integration.in_context_detector.add_few_shot_examples(examples)
        
        return integration 