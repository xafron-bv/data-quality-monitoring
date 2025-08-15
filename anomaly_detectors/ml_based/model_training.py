"""
ML model training module for anomaly detection.
"""

import json
import os
import random
import re
import sys
from datetime import datetime
from os import path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.error_injection import apply_error_rule, load_error_rules


def get_field_configs():
    """
    Get field-specific configurations for model training.
    Maps field names to their optimal model configurations.
    """
    return {
        # EXCELLENT PERFORMANCE (1.0 recall) - Use ACTUAL best found parameters
        'category': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 4},  # 1.0 recall
        'colour_code': {'model': 'sentence-transformers/all-mpnet-base-v2', 'epochs': 2},  # 1.0 recall
        'color_name': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 4},  # 1.0 recall

        # GOOD PERFORMANCE (0.8+ recall) - Use ACTUAL best found parameters
        'material': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.882 recall

        # MODERATE PERFORMANCE (0.4-0.5 recall) - Use ACTUAL best found parameters
        'ean': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.5 recall
        'long_description_nl': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.5 recall
        'customs_tariff_number': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 2},  # 0.5 recall
        'size': {'model': 'sentence-transformers/paraphrase-MiniLM-L6-v2', 'epochs': 8},  # 0.4 recall

        # POOR PERFORMANCE (0.0 recall) - Use ACTUAL best found, but may need further optimization
        'article_number': {'model': 'sentence-transformers/all-mpnet-base-v2', 'epochs': 5},  # 0.0 recall
        'description_short_1': {'model': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens', 'epochs': 5},  # 0.0 recall
        'product_name_en': {'model': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens', 'epochs': 6},  # 0.0 recall

        # DEFAULT CONFIGURATIONS (not tested yet)
        'care_instructions': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'season': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
    }


def preprocess_text(text):
    """
    Preprocess text to handle whitespace issues that caused missed detections.
    """
    if pd.isna(text):
        return text

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    return text


def create_improved_triplet_dataset(data_series, rules, field_name):
    """
    Create triplets for anomaly detection - corrupted data should be distant from clean data.
    All clean values are considered similar, anomalies are error-injected values.
    Args:
        data_series: Pandas series containing the data values
        rules: Error injection rules for the field
        field_name: The field name (used for logging)
    """
    print("Generating improved triplet dataset for anomaly detection...")

    # Preprocess data to handle whitespace issues
    preprocessed_data = data_series.dropna().apply(preprocess_text).astype(str)
    clean_texts = preprocessed_data.tolist()

    if not rules:
        print("Warning: No rules provided. Cannot generate triplet data.")
        return []

    if len(clean_texts) < 2:
        print("Warning: Need at least 2 clean texts to create triplets.")
        return []

    print(f"Working with {len(clean_texts)} clean texts from field '{field_name}'")

    triplets = []

    for anchor_text in clean_texts:
        # SIMPLIFIED LOGIC FOR ANOMALY DETECTION:
        # Anchor: Clean text (e.g., "red")
        # Positive: Any other clean text (e.g., "blue") - all clean values are similar
        # Negative: Error-injected anomaly (e.g., "chair", "134fdh") - true anomalies

        # Get positive candidates (any other clean text)
        positive_candidates = [text for text in clean_texts if text != anchor_text]

        if positive_candidates:
            positive_text = random.choice(positive_candidates)

            # Create negative example by applying error rules (corrupted variant)
            rule = random.choice(rules)
            negative_text = apply_error_rule(anchor_text, rule)

            # If rule didn't change the text, try another rule
            if negative_text == anchor_text and len(rules) > 1:
                negative_text = apply_error_rule(anchor_text, random.choice(rules))

            # Only add triplet if we actually created a corrupted version
            if negative_text != anchor_text:
                # Create InputExample for sentence transformers
                triplet = InputExample(
                    texts=[anchor_text, positive_text, negative_text],
                    label=0  # Not used in triplet loss
                )
                triplets.append(triplet)

    print(f"Created {len(triplets)} triplets for anomaly detection training.")
    print(f"Structure: Anchor (clean) -> Positive (clean) -> Negative (error-injected anomaly)")
    return triplets


def train_and_evaluate_similarity_model(df, field_name, column_name, rules, device, best_params, variation: str):
    """
    Train a sentence transformer model for RECALL-OPTIMIZED anomaly detection using triplet loss.
    Args:
        df: DataFrame containing the data
        field_name: The field name (used for model storage and rule loading)
        column_name: The column name in the CSV (used only for data access)
        rules: Error injection rules for the field
        device: Training device
        best_params: Hyperparameters for training
        variation: Required variation key for saving models per variation
    """
    if not variation:
        raise ValueError(f"Variation is required for training ML model for field '{field_name}'")
    print(f"\n🎯 Training RECALL-OPTIMIZED model for field '{field_name}' (column '{column_name}', variation '{variation}')")

    # Create results directory structure using field_name and variation
    models_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models', 'ml')
    model_results_dir = os.path.join(models_base_dir, f'{field_name.replace(" ", "_").lower()}', variation)
    checkpoints_dir = os.path.join(models_base_dir, 'checkpoints')

    # Ensure all directories exist
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Train final model with best parameters
    triplets = create_improved_triplet_dataset(df[column_name], rules, field_name)
    if not triplets:
        print(f"Could not create triplets for field '{field_name}' (column '{column_name}'). Skipping.")
        return None

    print(f"Training final model with RECALL-OPTIMIZED parameters...")
    print(f"Parameters: {best_params}")

    # Initialize sentence transformer model
    model = SentenceTransformer(best_params['model_name'])

    # Define loss function with optimized parameters
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=best_params['distance_metric'],
        triplet_margin=best_params['triplet_margin']
    )

    # Split data for evaluation
    split_idx = int(len(triplets) * 0.8)
    train_triplets = triplets[:split_idx]
    eval_triplets = triplets[split_idx:]

    # Create evaluation dataset
    if eval_triplets:
        evaluator = TripletEvaluator.from_input_examples(eval_triplets, name=f'{field_name}_final_evaluation')
        evaluation_steps = max(1, len(train_triplets) // best_params['batch_size'])
    else:
        evaluator = None
        evaluation_steps = 0

    # Train the model with organized output structure
    print(f"Training RECALL-OPTIMIZED anomaly detection model for field '{field_name}' with {len(train_triplets)} triplets...")
    print(f"Model outputs will be saved to: {model_results_dir}")

    model.fit(
        train_objectives=[(DataLoader(train_triplets, shuffle=True, batch_size=best_params['batch_size']), train_loss)],
        epochs=best_params['epochs'],
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        warmup_steps=50,
        output_path=model_results_dir,  # Save directly to organized results folder
        save_best_model=True,
        show_progress_bar=True
    )

    # Evaluate final performance
    if evaluator:
        print(f"\n--- RECALL-FOCUSED Performance for field '{field_name}' ---")
        final_results = evaluator(model, output_path=model_results_dir)
        if isinstance(final_results, dict):
            # Extract the accuracy score from the results dictionary
            for key, value in final_results.items():
                if 'accuracy' in key.lower():
                    print(f"  - {key}: {value:.4f}")
                    break
        else:
            print(f"  - Triplet Accuracy: {final_results:.4f}")

    # Test RECALL and PRECISION performance
    clean_texts = df[column_name].dropna().apply(preprocess_text).astype(str).tolist()
    print(f"\n💡 Testing final model performance with RECALL-OPTIMIZED threshold")
    recall_rate, precision_rate, f1_rate = test_recall_precision_performance(
        model, clean_texts, rules, field_name, threshold=0.6
    )

    # Compute and save reference centroid for production inference
    print(f"\n🔄 Computing reference centroid for production inference...")
    try:
        clean_training_texts = [text for text in clean_texts if text and len(text.strip()) > 0]
        if len(clean_training_texts) < 2:
            print(f"⚠️  Warning: Only {len(clean_training_texts)} clean texts available for centroid computation")

        clean_embeddings = model.encode(
            clean_training_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        reference_centroid = np.mean(clean_embeddings, axis=0)

        centroid_path = os.path.join(model_results_dir, "reference_centroid.npy")
        np.save(centroid_path, reference_centroid)

        centroid_metadata = {
            "num_samples": len(clean_training_texts),
            "embedding_dim": int(reference_centroid.shape[0]) if hasattr(reference_centroid, 'shape') else None,
            "created_at": datetime.now().isoformat(),
            "field_name": field_name,
            "column_name": column_name,
            "model_name": best_params['model_name'],
            "training_params": best_params,
            "variation": variation
        }
        metadata_path = os.path.join(model_results_dir, "centroid_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(centroid_metadata, f, indent=2, default=str)

        print(f"✅ Reference centroid saved at {centroid_path}")
        print(f"   📊 Based on {len(clean_training_texts)} samples")
    except Exception as e:
        print(f"⚠️  Warning: Could not compute/save reference centroid: {e}")

    print(f"✅ Training complete for field '{field_name}' (variation '{variation}')")
    return {
        'model_dir': model_results_dir,
        'detection_rate': recall_rate # Return recall rate as detection rate
    }


def demonstrate_similarity(model, data_series, field_name):
    """
    Demonstrate how the model calculates similarity between texts for anomaly detection
    Args:
        model: Trained sentence transformer model
        data_series: Pandas series containing the data values
        field_name: The field name (used for logging)
    """
    print(f"\n--- Anomaly Detection Demonstration for field '{field_name}' ---")

    # Get some sample texts and preprocess them
    sample_texts = data_series.dropna().apply(preprocess_text).astype(str).unique()[:10]

    if len(sample_texts) < 2:
        print("Not enough unique texts to demonstrate similarity.")
        return

    # Generate embeddings for all samples
    embeddings = model.encode(sample_texts)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Show top 3 most similar pairs (these should be clean-to-clean)
    print("Top 3 most similar pairs (clean texts):")
    pairs = []
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            pairs.append((i, j, similarity_matrix[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, (idx1, idx2, score) in enumerate(pairs[:3]):
        print(f"  {i+1}. '{sample_texts[idx1]}' ↔ '{sample_texts[idx2]}' (similarity: {score:.4f})")

    # Demonstrate anomaly detection with corrupted examples
    print("\n--- Anomaly Detection Test ---")

    # Load both error injection and anomaly injection rules for comprehensive training
    error_rules_dir = path.join(path.dirname(__file__), '..', '..', 'validators', 'error_injection_rules')
    anomaly_rules_dir = path.join(path.dirname(__file__), '..', 'anomaly_injection_rules')

    rule_files = {
        'category': 'category.json',
        'color_name': 'color_name.json',
        'season': 'season.json',
        'care_instructions': 'care_instructions.json',
        'article_number': 'article_number.json',
        'customs_tariff_number': 'customs_tariff_number.json',
        'description_short_1': 'description_short_1.json',
        'material': 'material.json',
        'product_name_en': 'product_name_en.json'
    }

    rule_file = rule_files.get(field_name, 'category.json')

    # Load both types of rules
    all_rules = []

    # Load error injection rules (format/validation anomalies)
    error_rule_path = os.path.join(error_rules_dir, rule_file)
    try:
        error_rules = load_error_rules(error_rule_path)
        all_rules.extend(error_rules)
        print(f"Loaded {len(error_rules)} error injection rules for demonstration")
    except FileNotFoundError:
        print(f"Error injection rules file '{error_rule_path}' not found.")

    # Load anomaly injection rules (semantic anomalies)
    anomaly_rule_path = os.path.join(anomaly_rules_dir, rule_file)
    try:
        anomaly_rules = load_anomaly_rules(anomaly_rule_path)
        # Convert anomaly rules to error rule format for compatibility
        converted_anomaly_rules = []
        for rule in anomaly_rules:
            converted_rule = {
                'rule_name': rule.get('rule_name', 'unknown'),
                'description': rule.get('description', ''),
                'operation': rule.get('operation', 'value_replacement'),
                'params': rule.get('params', {}),
                'conditions': rule.get('conditions', []),
                'is_anomaly_rule': True
            }
            converted_anomaly_rules.append(converted_rule)

        all_rules.extend(converted_anomaly_rules)
        print(f"Loaded {len(anomaly_rules)} anomaly injection rules for demonstration")
    except FileNotFoundError:
        print(f"Anomaly injection rules file '{anomaly_rule_path}' not found.")
    except Exception as e:
        print(f"Failed to load anomaly injection rules: {e}")

    # Fallback to simple rules if none found
    if not all_rules:
        print(f"No rules found. Creating simple corruption examples.")
        all_rules = [
            {'operation': 'random_noise'},
            {'operation': 'string_replace', 'params': {'find': 'e', 'replace': 'E'}}
        ]

    rules = all_rules

    if rules and len(sample_texts) > 0:
        # Take first 3 clean samples
        test_samples = sample_texts[:3]

        print("Testing RECALL-FOCUSED anomaly detection (lower similarity = better recall):")
        for i, clean_text in enumerate(test_samples):
            # Create corrupted version
            rule = random.choice(rules)
            corrupted_text = apply_error_rule(clean_text, rule)

            if corrupted_text != clean_text:
                # Get embeddings
                clean_embedding = model.encode([clean_text])
                corrupted_embedding = model.encode([corrupted_text])

                # Calculate similarity
                similarity = cosine_similarity(clean_embedding, corrupted_embedding)[0][0]

                print(f"  {i+1}. Clean: '{clean_text}' vs Anomaly: '{corrupted_text}'")
                # Use recall-focused threshold (0.6 instead of 0.8)
                print(f"     Similarity: {similarity:.4f} {'✅ DETECTED (good recall)' if similarity < 0.6 else '⚠️ MISSED (need lower threshold)'}")

    # Show embeddings dimensionality
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Total samples processed: {len(sample_texts)}")
    print(f"🎯 RECALL-FOCUSED: All clean values should be similar, anomalies should be distant")


def test_recall_precision_performance(model, clean_texts, rules, field_name, threshold=0.6):
    """
    Test the trained model's recall and precision performance on anomaly detection.
    Args:
        model: Trained sentence transformer model
        clean_texts: List of clean text samples
        rules: Error injection rules for the field
        field_name: The field name (used for logging)
        threshold: Anomaly detection threshold
    Returns (recall_rate, precision_rate, f1_rate).
    """
    print(f"\n🎯 Testing RECALL and PRECISION performance for field '{field_name}'...")

    if not rules or len(clean_texts) < 2:
        print("Not enough data to test performance.")
        return 0.0, 0.0, 0.0

    # Preprocess clean texts
    clean_texts = [preprocess_text(text) for text in clean_texts]

    # Create test set
    test_clean = clean_texts[:min(20, len(clean_texts))]  # Limit to 20 for speed

    print(f"Testing with {len(test_clean)} clean samples and threshold {threshold}")

    # Test recall: how many actual anomalies are detected
    detected_anomalies = 0
    total_anomalies = 0

    print("\n📊 RECALL TEST (Corrupted vs Clean):")
    for i, clean_text in enumerate(test_clean):
        # Create corrupted version
        rule = random.choice(rules)
        corrupted_text = apply_error_rule(clean_text, rule)

        # Preprocess corrupted text as well
        corrupted_text = preprocess_text(corrupted_text)

        if corrupted_text != clean_text:
            # Get embeddings
            clean_embedding = model.encode([clean_text])
            corrupted_embedding = model.encode([corrupted_text])

            # Calculate similarity
            similarity = cosine_similarity(clean_embedding, corrupted_embedding)[0][0]

            # Check if anomaly was detected (similarity below threshold)
            if similarity < threshold:
                detected_anomalies += 1
                status = "✅ DETECTED"
            else:
                status = "❌ MISSED"

            print(f"  {i+1}. '{clean_text}' vs '{corrupted_text}' → {similarity:.3f} {status}")
            total_anomalies += 1

    # Test precision: how many flagged items are actually anomalies
    # Test clean-to-clean similarities to see false positives
    clean_pairs_flagged = 0
    total_clean_pairs = 0

    print("\n📊 PRECISION TEST (Clean vs Clean - should NOT be flagged):")
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
                status = "❌ FALSE POSITIVE"
            else:
                status = "✅ CORRECTLY IGNORED"

            print(f"  '{clean_text1}' vs '{clean_text2}' → {similarity:.3f} {status}")
            total_clean_pairs += 1

    # Calculate metrics
    recall = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0
    precision = detected_anomalies / (detected_anomalies + clean_pairs_flagged) if (detected_anomalies + clean_pairs_flagged) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n📈 FINAL PERFORMANCE METRICS:")
    print(f"  🎯 RECALL: {recall:.4f} ({recall*100:.1f}%) - {detected_anomalies}/{total_anomalies} anomalies detected")
    print(f"  🎯 PRECISION: {precision:.4f} ({precision*100:.1f}%) - {detected_anomalies}/{detected_anomalies + clean_pairs_flagged} flagged items were actual anomalies")
    print(f"  🎯 F1 SCORE: {f1_score:.4f} ({f1_score*100:.1f}%)")

    if precision >= 0.3:
        print(f"  ✅ PRECISION CONSTRAINT MET: {precision*100:.1f}% ≥ 30%")
    else:
        print(f"  ❌ PRECISION CONSTRAINT NOT MET: {precision*100:.1f}% < 30%")

    return recall, precision, f1_score


def test_anomaly_detection(model, clean_texts, rules, field_name, threshold=0.6):
    """
    Test the trained model's ability to detect anomalies with RECALL FOCUS.
    Args:
        model: Trained sentence transformer model
        clean_texts: List of clean text samples
        rules: Error injection rules for the field
        field_name: The field name (used for logging)
        threshold: Anomaly detection threshold (default 0.6 for recall focus)
    Uses lower threshold (0.6) to catch more anomalies.
    """
    print(f"\n🎯 Testing RECALL-FOCUSED anomaly detection for field '{field_name}'...")

    if not rules or len(clean_texts) < 2:
        print("Not enough data to test anomaly detection.")
        return 0.0

    # Preprocess clean texts
    clean_texts = [preprocess_text(text) for text in clean_texts]

    # Create test set
    test_clean = clean_texts[:min(20, len(clean_texts))]  # Limit to 20 for speed
    detected_anomalies = 0
    total_tests = 0

    print(f"Testing with {len(test_clean)} clean samples and RECALL-OPTIMIZED threshold {threshold}")

    for clean_text in test_clean:
        # Create corrupted version
        rule = random.choice(rules)
        corrupted_text = apply_error_rule(clean_text, rule)

        # Preprocess corrupted text as well
        corrupted_text = preprocess_text(corrupted_text)

        if corrupted_text != clean_text:
            # Get embeddings
            clean_embedding = model.encode([clean_text])
            corrupted_embedding = model.encode([corrupted_text])

            # Calculate similarity
            similarity = cosine_similarity(clean_embedding, corrupted_embedding)[0][0]

            # Check if anomaly was detected (similarity below threshold)
            if similarity < threshold:
                detected_anomalies += 1
                status = "✅ DETECTED"
            else:
                status = "❌ MISSED"

            print(f"  Clean: '{clean_text}' vs Corrupted: '{corrupted_text}' -> {similarity:.3f} {status}")
            total_tests += 1

    if total_tests > 0:
        detection_rate = (detected_anomalies / total_tests) * 100
        print(f"\n🎯 RECALL-FOCUSED Anomaly Detection Rate: {detection_rate:.1f}% ({detected_anomalies}/{total_tests})")
        print(f"💡 Strategy: Prioritize catching anomalies over avoiding false positives")
        return detection_rate
    else:
        print("No anomalies could be generated for testing.")
        return 0.0


def setup_results_directory_structure():
    """
    Create the organized directory structure for all results and outputs.
    """
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'data', 'models', 'ml')

    # Create main directories
    directories = [
        models_dir,
        os.path.join(models_dir, 'checkpoints'),
        os.path.join(models_dir, 'summary'),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"📁 Model directory structure created:")
    print(f"  - data/models/ml/")
    print(f"    ├── <field>/<variation>/  (trained models)")
    print(f"    ├── summary/              (HP search results, summaries)")
    print(f"    └── checkpoints/          (temporary training checkpoints)")

    return models_dir
