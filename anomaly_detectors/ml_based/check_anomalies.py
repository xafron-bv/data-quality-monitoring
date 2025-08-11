import json
import os
import sys
from os import path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from anomaly_detectors.ml_based.gpu_utils import get_optimal_batch_size, get_optimal_device, print_device_info
from anomaly_detectors.ml_based.model_training import preprocess_text
from common.field_column_map import get_field_to_column_map

# Global model cache to avoid reloading models
_model_cache = {}

def load_model_for_field(field_name, models_dir=None, use_gpu=True, variation=None):
    """
    Given a field name, load the corresponding model and return the model, column name, and reference centroid.
    Uses caching to avoid reloading the same model multiple times.

    Args:
        field_name: Name of the field to load model for
        models_dir: Directory containing the trained models
        use_gpu: Whether to use GPU acceleration if available
        variation: Variation key to load variant-specific model directory (required)

    Returns:
        tuple: (model, column_name, reference_centroid)

    Raises:
        FileNotFoundError: If reference centroid is not found
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models', 'ml')

    # Create cache key
    cache_key = (field_name, models_dir, use_gpu, variation)

    # Check if model is already cached
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    field_to_column = get_field_to_column_map()
    if field_name not in field_to_column:
        raise ValueError(f"Field '{field_name}' not found in field-to-column map.")
    column_name = field_to_column[field_name]

    # Require variation
    if not variation:
        raise ValueError(f"Variation is required for field '{field_name}' when loading ML model")

    trained_root = models_dir
    field_root = os.path.join(trained_root, f'{field_name.replace(" ", "_").lower()}')
    variant_dir = os.path.join(field_root, variation)

    if os.path.isdir(variant_dir):
        model_dir = variant_dir
    else:
        raise FileNotFoundError(
            f"Model directory for field '{field_name}' variation '{variation}' not found at {variant_dir}"
        )

    # Determine device to use
    device = get_optimal_device(use_gpu)
    print_device_info(device, f"field '{field_name}'")

    # Load model with GPU support
    model = SentenceTransformer(model_dir, device=device)

    # Load reference centroid (required)
    centroid_path = os.path.join(model_dir, "reference_centroid.npy")
    metadata_path = os.path.join(model_dir, "centroid_metadata.json")

    if not os.path.exists(centroid_path):
        raise FileNotFoundError(f"Reference centroid not found at {centroid_path}. Please retrain the model to generate centroid.")

    try:
        reference_centroid = np.load(centroid_path)
        print(f"✅ Loaded reference centroid (shape: {reference_centroid.shape})")

        # Load and display metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   📊 Based on {metadata.get('num_samples', 'unknown')} training samples")
            print(f"   📅 Created: {metadata.get('created_at', 'unknown')}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load reference centroid from {centroid_path}: {e}")

    # Cache the loaded model and centroid
    result = (model, column_name, reference_centroid)
    _model_cache[cache_key] = result

    return result


def check_anomalies(model, values, threshold=0.6, reference_centroid=None):
    """
    Check anomalies using pre-computed reference centroid.
    This is the production-ready approach that works for both single values and batches.

    Args:
        model: SentenceTransformer model
        values: List of values to check (or single value in a list)
        threshold: Similarity threshold for anomaly detection
        reference_centroid: Pre-computed reference centroid (required)

    Returns:
        List of result dictionaries with 'value', 'is_anomaly', 'probability_of_correctness'
    """
    if reference_centroid is None:
        raise ValueError("Reference centroid is required. Use load_model_for_field to get the centroid.")

    if not isinstance(values, list):
        values = [values]

    results = []

    # Preprocess values
    processed_values = []
    for value in values:
        value_prep = preprocess_text(value)
        value_str = str(value_prep) if value_prep is not None else ""
        processed_values.append(value_str)

    # Determine optimal batch size based on device
    device_str = str(model.device) if hasattr(model, 'device') else 'cpu'
    batch_size = min(get_optimal_batch_size(device_str), len(processed_values))

    # Encode all values
    embeddings = model.encode(processed_values,
                             batch_size=batch_size,
                             show_progress_bar=False,
                             convert_to_numpy=True)

    # Compute similarities to reference centroid
    centroid_sims = cosine_similarity(embeddings, reference_centroid.reshape(1, -1)).flatten()

    # Create results
    for i, value in enumerate(values):
        similarity = float(centroid_sims[i])
        is_anomaly = similarity < threshold

        results.append({
            'value': value,
            'is_anomaly': is_anomaly,
            'probability_of_correctness': similarity
        })

    return results
