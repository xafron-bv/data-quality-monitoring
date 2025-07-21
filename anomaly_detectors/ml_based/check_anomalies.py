import numpy as np
from os import path
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model_training import preprocess_text

# Import field-to-column mapping
from field_column_map import get_field_to_column_map

def load_model_for_field(field_name, results_dir=path.join('..', 'results')):
    """
    Given a field name, load the corresponding model and return the model and the mapped column name.
    """
    field_to_column = get_field_to_column_map()
    if field_name not in field_to_column:
        raise ValueError(f"Field '{field_name}' not found in field-to-column map.")
    column_name = field_to_column[field_name]
    model_dir = os.path.join(results_dir, f'results_{field_name.replace(" ", "_").lower()}')
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found for field '{field_name}' (column '{column_name}'): {model_dir}")
    return SentenceTransformer(model_dir), column_name





def check_anomalies(model, values, threshold=0.6):
    """
    Check anomalies using the model's learned representation.
    
    This approach uses the fact that the model was trained with triplet loss to distinguish
    between normal and anomalous texts. We compute embeddings and use statistical measures
    to identify outliers based on centroid distance.
    """
    results = []
    
    # Get embeddings for all values
    processed_values = []
    embeddings = []
    
    for value in values:
        value_prep = preprocess_text(value)
        value_str = str(value_prep) if value_prep is not None else ""
        processed_values.append(value_str)
    
    # Encode all values at once for efficiency
    embeddings = model.encode(processed_values)
    
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
