import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model_training import preprocess_text

# Import rule-to-column mapping
from rule_column_map import get_rule_to_column_map

def load_model_for_rule(rule_name, results_dir='../results'):
    """
    Given a rule name, load the corresponding model and return the model and the mapped column name.
    """
    rule_to_column = get_rule_to_column_map()
    if rule_name not in rule_to_column:
        raise ValueError(f"Rule '{rule_name}' not found in rule-to-column map.")
    column_name = rule_to_column[rule_name]
    model_dir = os.path.join(results_dir, f'results_{column_name.replace(" ", "_").lower()}')
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found for rule '{rule_name}' (column '{column_name}'): {model_dir}")
    return SentenceTransformer(model_dir), column_name


def get_reference_clean_values(df, column_name, max_samples=100):
    """
    Get reference clean values from the provided dataframe (should be the original clean training data).
    """
    clean_texts = df[column_name].dropna().apply(preprocess_text).astype(str).unique()
    return clean_texts[:max_samples]


def get_reference_clean_values_from_file(clean_csv_path, column_name, max_samples=100):
    """
    Load reference clean values from the original training dataset file.
    """
    import pandas as pd
    clean_df = pd.read_csv(clean_csv_path)
    return get_reference_clean_values(clean_df, column_name, max_samples)


def check_anomalies_with_references(model, values, clean_values, threshold=0.6):
    """
    Check anomalies by comparing against clean reference values.
    This approach requires clean reference data.
    """
    results = []
    # Ensure all clean values are strings
    clean_embs = model.encode([str(v) for v in clean_values])
    for value in values:
        value_prep = preprocess_text(value)
        value_str = str(value_prep) if value_prep is not None else ""
        value_emb = model.encode([value_str])
        sims = cosine_similarity(value_emb, clean_embs)[0]
        max_sim = float(np.max(sims))
        is_anom = max_sim < threshold
        results.append({
            'value': value,
            'is_anomaly': is_anom,
            'probability_of_correctness': max_sim
        })
    return results


def check_anomalies(model, values, threshold=0.6):
    """
    Check anomalies using the model's learned representation without requiring clean references.
    
    This approach uses the fact that the model was trained with triplet loss to distinguish
    between clean and anomalous texts. We compute embeddings and use statistical measures
    to identify outliers.
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
