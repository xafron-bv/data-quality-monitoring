import numpy as np
from os import path
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model_training import preprocess_text
from gpu_utils import get_optimal_device, print_device_info, get_optimal_batch_size

# Import field-to-column mapping
from field_column_map import get_field_to_column_map

# Global model cache to avoid reloading models
_model_cache = {}

def load_model_for_field(field_name, results_dir=path.join('..', 'results'), use_gpu=True):
    """
    Given a field name, load the corresponding model and return the model and the mapped column name.
    Uses caching to avoid reloading the same model multiple times.
    
    Args:
        field_name: Name of the field to load model for
        results_dir: Directory containing the trained models
        use_gpu: Whether to use GPU acceleration if available
    """
    # Create cache key
    cache_key = (field_name, results_dir, use_gpu)
    
    # Check if model is already cached
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    field_to_column = get_field_to_column_map()
    if field_name not in field_to_column:
        raise ValueError(f"Field '{field_name}' not found in field-to-column map.")
    column_name = field_to_column[field_name]
    model_dir = os.path.join(results_dir, f'results_{field_name.replace(" ", "_").lower()}')
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found for field '{field_name}' (column '{column_name}'): {model_dir}")
    
    # Determine device to use
    device = get_optimal_device(use_gpu)
    print_device_info(device, f"field '{field_name}'")
    
    # Load model with GPU support
    model = SentenceTransformer(model_dir, device=device)
    
    # Cache the result
    result = (model, column_name)
    _model_cache[cache_key] = result
    
    return result





def check_anomalies(model, values, threshold=0.6, batch_size=None):
    """
    Check anomalies using the model's learned representation with GPU acceleration.
    
    This approach uses the fact that the model was trained with triplet loss to distinguish
    between normal and anomalous texts. We compute embeddings and use statistical measures
    to identify outliers based on centroid distance.
    
    Args:
        model: SentenceTransformer model
        values: List of values to check
        threshold: Similarity threshold for anomaly detection
        batch_size: Batch size for encoding (larger for GPU, smaller for CPU)
    """
    results = []
    
    # Get embeddings for all values
    processed_values = []
    
    for value in values:
        value_prep = preprocess_text(value)
        value_str = str(value_prep) if value_prep is not None else ""
        processed_values.append(value_str)
    
    # Determine optimal batch size based on device
    if batch_size is None:
        # Use GPU utility to determine optimal batch size
        device_str = str(model.device) if hasattr(model, 'device') else 'cpu'
        batch_size = get_optimal_batch_size(device_str, default_gpu=512, default_cpu=32)
        batch_size = min(batch_size, len(processed_values))
    
    # Encode all values at once for efficiency with specified batch size
    embeddings = model.encode(processed_values, 
                             batch_size=batch_size,
                             show_progress_bar=False,
                             convert_to_numpy=True)
    
    # Compute centroid of all embeddings as a baseline "normal" representation
    centroid = np.mean(embeddings, axis=0)
    
    # Compute similarities to centroid
    centroid_sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # Also compute pairwise similarities to identify outliers
    pairwise_sims = cosine_similarity(embeddings)
    
    for i, (value, embedding) in enumerate(zip(values, embeddings)):
        # Similarity to centroid
        centroid_sim = float(centroid_sims[i])
        
        # Average similarity to all other values (excluding self)
        other_sims = np.concatenate([pairwise_sims[i][:i], pairwise_sims[i][i+1:]])
        avg_sim_to_others = float(np.mean(other_sims)) if len(other_sims) > 0 else 1.0
        
        # Use the minimum of the two similarity measures
        # This captures both global outliers (far from centroid) and local outliers (dissimilar to neighbors)
        final_similarity = min(centroid_sim, avg_sim_to_others)
        
        is_anom = final_similarity < threshold
        
        results.append({
            'value': value,
            'is_anomaly': is_anom,
            'probability_of_correctness': final_similarity
        })
    
    return results
