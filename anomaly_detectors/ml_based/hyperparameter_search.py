"""
Hyperparameter search for ML anomaly detection models.
"""

import glob
import json
import os
import random
import re
import sys
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from anomaly_detectors.ml_based.model_training import create_improved_triplet_dataset, preprocess_text
from common.anomaly_injection import load_anomaly_rules


def get_optimal_parameters(field_name, fallback_model_name, fallback_epochs):
    """
    Get RECALL-OPTIMIZED parameters for each field based on ACTUAL hyperparameter search results.
    Now requires optimal_params.json to be present and include the field.
    """
    params_path = os.path.join(os.path.dirname(__file__), "optimal_params.json")
    try:
        with open(params_path, "r") as f:
            optimal_params = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"optimal_params.json is required but could not be loaded: {e}")

    def parse_distance_metric(metric_str):
        mapping = {
            "COSINE": losses.TripletDistanceMetric.COSINE,
            "EUCLIDEAN": losses.TripletDistanceMetric.EUCLIDEAN,
            "MANHATTAN": losses.TripletDistanceMetric.MANHATTAN,
        }
        if metric_str not in mapping:
            raise ValueError(f"Unknown distance metric: {metric_str}")
        return mapping[metric_str]

    if field_name not in optimal_params:
        raise KeyError(f"No optimal parameters found for field '{field_name}' in {params_path}")

    params = optimal_params[field_name]
    params["distance_metric"] = parse_distance_metric(params["distance_metric"])
    return params


def train_with_params(df, field_name, column_name, rules, params):
    """
    Train model with specific hyperparameters and return RECALL and PRECISION scores.
    Args:
        df: DataFrame containing the data
        field_name: The field name (used for model identification)
        column_name: The column name in the CSV (used only for data access)
        rules: Error injection rules for the field
        params: Hyperparameters for training
    """

    triplets = create_improved_triplet_dataset(df[column_name], rules, field_name)
    if not triplets:
        return -1, -1, -1

    # Get clean texts for evaluation
    clean_texts = df[column_name].dropna().apply(preprocess_text).astype(str).tolist()
    if len(clean_texts) < 5:
        return -1, -1, -1

    # Initialize model
    model = SentenceTransformer(params['model_name'])

    # Split data
    split_idx = int(len(triplets) * 0.8)
    train_triplets = triplets[:split_idx]
    eval_triplets = triplets[split_idx:]

    if not eval_triplets:
        return -1, -1, -1

    # Create loss function with hyperparameters
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=params['distance_metric'],
        triplet_margin=params['triplet_margin']
    )

    # Ensure checkpoints directory exists for temporary training during HP search
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Train model WITHOUT evaluator and WITHOUT saving (we'll evaluate separately)
    model.fit(
        train_objectives=[(DataLoader(train_triplets, shuffle=True, batch_size=params['batch_size']), train_loss)],
        epochs=params['epochs'],
        evaluator=None,  # No evaluator during HP search
        evaluation_steps=0,
        warmup_steps=50,
        output_path=None,  # Don't save during HP search to avoid clutter
        save_best_model=False,
        show_progress_bar=False
    )

    # Evaluate RECALL and PRECISION performance
    try:
        recall_score, precision_score, f1_score = evaluate_recall_and_precision_performance(model, clean_texts, rules, field_name)
        return recall_score, precision_score, f1_score
    except Exception as e:
        print(f"   ❌ Performance evaluation failed: {e}")
        return -1, -1, -1


def evaluate_recall_and_precision_performance(model, clean_texts, rules, field_name, num_samples=20):
    """
    Evaluate model's recall and precision performance on anomaly detection.
    Args:
        model: Trained sentence transformer model
        clean_texts: List of clean text samples
        rules: Error injection rules for the field
        field_name: The field name (used for logging)
        num_samples: Number of samples to use for evaluation
    Returns (recall_score, precision_score, f1_score).
    """

    if not rules or len(clean_texts) < 4:
        return 0.0, 0.0, 0.0

    # Take sample of clean texts for evaluation
    test_clean = clean_texts[:min(num_samples, len(clean_texts))]

    # Use recall-optimized threshold
    threshold = 0.6

    # Test recall: how many actual anomalies are detected
    detected_anomalies = 0
    total_anomalies = 0

    for clean_text in test_clean:
        # Create corrupted version
        rule = random.choice(rules)
        corrupted_text = apply_error_rule(clean_text, rule)

        # Preprocess corrupted text
        corrupted_text = preprocess_text(corrupted_text)

        if corrupted_text != clean_text:
            # Get embeddings
            clean_embedding = model.encode([clean_text])
            corrupted_embedding = model.encode([corrupted_text])

            # Calculate similarity
            similarity = cosine_similarity(clean_embedding, corrupted_embedding)[0][0]

            # Check if anomaly was detected
            if similarity < threshold:
                detected_anomalies += 1

            total_anomalies += 1

    # Test precision: how many flagged items are actually anomalies
    # Test clean-to-clean similarities to see false positives
    clean_pairs_flagged = 0
    total_clean_pairs = 0

    # Test random pairs of clean texts
    for i in range(min(10, len(test_clean))):
        for j in range(i+1, min(i+6, len(test_clean))):  # Limit pairs to avoid too many tests
            clean_text1 = test_clean[i]
            clean_text2 = test_clean[j]

            # Get embeddings
            clean_embedding1 = model.encode([clean_text1])
            clean_embedding2 = model.encode([clean_text2])

            # Calculate similarity
            similarity = cosine_similarity(clean_embedding1, clean_embedding2)[0][0]

            # Check if clean pair was flagged as anomalous (false positive)
            if similarity < threshold:
                clean_pairs_flagged += 1

            total_clean_pairs += 1

    # Calculate metrics
    recall = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0

    # Precision = True Positives / (True Positives + False Positives)
    # True Positives = detected_anomalies
    # False Positives = clean_pairs_flagged
    precision = detected_anomalies / (detected_anomalies + clean_pairs_flagged) if (detected_anomalies + clean_pairs_flagged) > 0 else 0.0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return recall, precision, f1_score


def random_hyperparameter_search(df, field_name, column_name, rules, device, num_trials=15):
    """
    Perform RECALL-FOCUSED hyperparameter search with PRECISION constraint (min 30%).
    Args:
        df: DataFrame containing the data
        field_name: The field name (used for saving results and logging)
        column_name: The column name in the CSV (used only for data access)
        rules: Error injection rules for the field
        device: Training device
        num_trials: Number of hyperparameter trials
    """
    print(f"\n🎯 Starting RECALL-FOCUSED hyperparameter search for field '{field_name}' with {num_trials} trials...")
    print(f"💡 Constraint: Precision must be at least 30% to be considered valid")

    # Load hyperparameter search space from JSON file
    hp_space_path = os.path.join(os.path.dirname(__file__), "hyperparameter_search_space.json")
    try:
        with open(hp_space_path, "r") as f:
            hp_search_space = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load hyperparameter_search_space.json: {e}")
        return None, None, None, None, []

    # Convert string distance_metric to actual TripletDistanceMetric for the search space
    def parse_distance_metric(metric_str):
        if metric_str == "COSINE":
            return losses.TripletDistanceMetric.COSINE
        elif metric_str == "EUCLIDEAN":
            return losses.TripletDistanceMetric.EUCLIDEAN
        elif metric_str == "MANHATTAN":
            return losses.TripletDistanceMetric.MANHATTAN
        else:
            return losses.TripletDistanceMetric.COSINE

    best_recall = -1
    best_precision = -1
    best_f1 = -1
    best_params = None
    results = []

    # Track performance trends
    model_scores = {}
    margin_scores = {}
    distance_scores = {}

    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")

        # Sample random hyperparameters
        params = {}
        for param_name, values in hp_search_space.items():
            val = random.choice(values)
            if param_name == "distance_metric":
                val = parse_distance_metric(val)
            params[param_name] = val

        print(f"Testing parameters: {params}")

        try:
            # Train with current parameters and get RECALL, PRECISION, F1 scores
            recall_score, precision_score, f1_score = train_with_params(df, field_name, column_name, rules, params)

            # Only consider results with minimum 30% precision
            if precision_score >= 0.3:
                results.append((params.copy(), recall_score, precision_score, f1_score))

                # Track performance by parameter
                model_name = params['model_name'].split('/')[-1]
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append((recall_score, precision_score, f1_score))

                margin = params['triplet_margin']
                if margin not in margin_scores:
                    margin_scores[margin] = []
                margin_scores[margin].append((recall_score, precision_score, f1_score))

                distance = str(params['distance_metric']).split('.')[-1]
                if distance not in distance_scores:
                    distance_scores[distance] = []
                distance_scores[distance].append((recall_score, precision_score, f1_score))

                # Update best based on recall (primary) with precision constraint
                if recall_score > best_recall:
                    best_recall = recall_score
                    best_precision = precision_score
                    best_f1 = f1_score
                    best_params = params.copy()
                    print(f"🎯 New best RECALL score: {recall_score:.4f} ({recall_score*100:.1f}%)")
                    print(f"   Precision: {precision_score:.4f} ({precision_score*100:.1f}%)")
                    print(f"   F1 Score: {f1_score:.4f} ({f1_score*100:.1f}%)")
            else:
                print(f"   ❌ Rejected: Precision {precision_score:.4f} ({precision_score*100:.1f}%) < 30% threshold")
                results.append((params.copy(), recall_score, precision_score, f1_score))

        except Exception as e:
            print(f"❌ Trial {trial + 1} failed: {e}")
            results.append((params.copy(), -1, -1, -1))

    # Print analysis of parameter performance
    print(f"\n📊 RECALL-FOCUSED Parameter Performance Analysis (Precision ≥ 30%):")
    print(f"{'='*60}")

    print("\n🤖 Model Performance:")
    for model, scores in model_scores.items():
        valid_scores = [s for s in scores if s[0] > 0 and s[1] >= 0.3]
        if valid_scores:
            avg_recall = sum(s[0] for s in valid_scores) / len(valid_scores)
            avg_precision = sum(s[1] for s in valid_scores) / len(valid_scores)
            avg_f1 = sum(s[2] for s in valid_scores) / len(valid_scores)
            print(f"  {model}: Recall={avg_recall:.4f} ({avg_recall*100:.1f}%), Precision={avg_precision:.4f} ({avg_precision*100:.1f}%), F1={avg_f1:.4f}")

    print("\n📏 Triplet Margin Performance:")
    for margin, scores in sorted(margin_scores.items()):
        valid_scores = [s for s in scores if s[0] > 0 and s[1] >= 0.3]
        if valid_scores:
            avg_recall = sum(s[0] for s in valid_scores) / len(valid_scores)
            avg_precision = sum(s[1] for s in valid_scores) / len(valid_scores)
            avg_f1 = sum(s[2] for s in valid_scores) / len(valid_scores)
            print(f"  {margin}: Recall={avg_recall:.4f} ({avg_recall*100:.1f}%), Precision={avg_precision:.4f} ({avg_precision*100:.1f}%), F1={avg_f1:.4f}")

    print("\n📐 Distance Metric Performance:")
    for distance, scores in distance_scores.items():
        valid_scores = [s for s in scores if s[0] > 0 and s[1] >= 0.3]
        if valid_scores:
            avg_recall = sum(s[0] for s in valid_scores) / len(valid_scores)
            avg_precision = sum(s[1] for s in valid_scores) / len(valid_scores)
            avg_f1 = sum(s[2] for s in valid_scores) / len(valid_scores)
            print(f"  {distance}: Recall={avg_recall:.4f} ({avg_recall*100:.1f}%), Precision={avg_precision:.4f} ({avg_precision*100:.1f}%), F1={avg_f1:.4f}")

    if best_params:
        print(f"\n🏆 Best RECALL-OPTIMIZED parameters (Precision ≥ 30%):")
        print(f"RECALL Score: {best_recall:.4f} ({best_recall*100:.1f}%)")
        print(f"PRECISION Score: {best_precision:.4f} ({best_precision*100:.1f}%)")
        print(f"F1 Score: {best_f1:.4f} ({best_f1*100:.1f}%)")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print(f"\n❌ No valid parameters found with precision ≥ 30%")
        print(f"💡 Consider relaxing precision constraint or adjusting hyperparameter space")

    # Save hyperparameter search results to file
    results_dir = os.path.join(os.path.dirname(__file__), 'models', 'summary')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"hp_search_results_{field_name.replace(' ', '_').lower()}.json")
    hp_results = {
        'field_name': field_name,
        'column_name': column_name,  # Store both for reference
        'best_params': best_params,
        'best_recall': best_recall,
        'best_precision': best_precision,
        'best_f1': best_f1,
        'all_results': [(params, recall, precision, f1) for params, recall, precision, f1 in results],
        'performance_analysis': {
            'model_scores': {model: {
                'avg_recall': sum(s[0] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_precision': sum(s[1] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_f1': sum(s[2] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0
            } for model, scores in model_scores.items()},
            'margin_scores': {str(margin): {
                'avg_recall': sum(s[0] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_precision': sum(s[1] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_f1': sum(s[2] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0
            } for margin, scores in margin_scores.items()},
            'distance_scores': {distance: {
                'avg_recall': sum(s[0] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_precision': sum(s[1] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0,
                'avg_f1': sum(s[2] for s in scores if s[0] > 0 and s[1] >= 0.3) / len([s for s in scores if s[0] > 0 and s[1] >= 0.3]) if any(s[0] > 0 and s[1] >= 0.3 for s in scores) else 0
            } for distance, scores in distance_scores.items()}
        }
    }

    with open(results_file, 'w') as f:
        json.dump(hp_results, f, indent=2, default=str)

    print(f"📁 Hyperparameter search results saved to: {results_file}")

    return best_params, best_recall, best_precision, best_f1, results


def save_aggregated_hp_results():
    """
    Read all individual hyperparameter search results and create an aggregated summary.
    """

    # Look for HP search results in the models/summary directory
    results_dir = os.path.join(os.path.dirname(__file__), 'models', 'summary')
    hp_files = glob.glob(os.path.join(results_dir, "hp_search_results_*.json"))

    if not hp_files:
        print("No hyperparameter search results found.")
        return

    aggregated_results = {}
    all_best_params = {}

    for hp_file in hp_files:
        try:
            with open(hp_file, 'r') as f:
                data = json.load(f)

                # Handle both old and new format
                if 'field_name' in data:
                    # New format with field names
                    field_name = data['field_name']
                    column_name = data.get('column_name', field_name)  # fallback to field_name if not present
                    key = field_name  # Use field name as the key
                elif 'column' in data:
                    # Old format with column names
                    column_name = data['column']
                    field_name = column_name  # fallback when converting from old format
                    key = column_name  # Use column name for backward compatibility
                else:
                    print(f"Invalid format in {hp_file}: missing field_name or column key")
                    continue

                # Handle both old and new format
                if 'best_recall' in data:
                    # New format with recall, precision, f1
                    aggregated_results[key] = {
                        'field_name': field_name,
                        'column_name': column_name,
                        'best_recall': data['best_recall'],
                        'best_precision': data['best_precision'],
                        'best_f1': data['best_f1'],
                        'best_params': data['best_params'],
                        'performance_analysis': data['performance_analysis']
                    }
                else:
                    # Old format with just best_score
                    aggregated_results[key] = {
                        'field_name': field_name,
                        'column_name': column_name,
                        'best_recall': data.get('best_score', 0),
                        'best_precision': 0,  # Unknown in old format
                        'best_f1': 0,  # Unknown in old format
                        'best_params': data['best_params'],
                        'performance_analysis': data['performance_analysis']
                    }

                all_best_params[key] = data['best_params']
        except Exception as e:
            print(f"Error reading {hp_file}: {e}")

    # Save aggregated results in the summary folder
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_fields': len(aggregated_results),
        'field_results': aggregated_results,
        'recommended_configs': all_best_params
    }

    summary_file = os.path.join(results_dir, 'hp_search_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📋 HYPERPARAMETER SEARCH SUMMARY (with Precision Constraint):")
    print(f"{'='*60}")
    print(f"Total columns tested: {len(aggregated_results)}")
    print(f"\nRanked by RECALL Score (Precision ≥ 30%):")

    # Sort by recall score
    sorted_results = sorted(aggregated_results.items(), key=lambda x: x[1]['best_recall'], reverse=True)

    for column, result in sorted_results:
        recall = result['best_recall']
        precision = result['best_precision']
        f1 = result['best_f1']
        model = result['best_params']['model_name'].split('/')[-1] if result['best_params'] else 'Unknown'

        if result['best_params']:
            margin = result['best_params']['triplet_margin']
            epochs = result['best_params']['epochs']
            batch_size = result['best_params']['batch_size']

            print(f"  {column}:")
            print(f"    📊 Recall: {recall:.4f} ({recall*100:.1f}%)")
            print(f"    📊 Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"    📊 F1 Score: {f1:.4f} ({f1*100:.1f}%)")
            print(f"    🤖 Model: {model}")
            print(f"    ⚙️  Margin: {margin}, Epochs: {epochs}, Batch: {batch_size}")

            if precision >= 0.3:
                print(f"    ✅ Precision constraint met")
            else:
                print(f"    ❌ Precision constraint not met")
        else:
            print(f"  {column}: No valid parameters found")

    print(f"\n📁 Full summary saved to: {summary_file}")
    return summary
