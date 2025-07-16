"""
Anomaly Detection using Sentence Transformers and Triplet Loss

Key Changes for Anomaly Detection:
1. FLIPPED TRIPLET LOGIC:
   - Anchor: Clean text (e.g., "red")
   - Positive: Other clean text from same semantic group (e.g., "blue")
   - Negative: Corrupted text (e.g., "re d") - should be far from anchor

2. OPTIMIZED HYPERPARAMETERS:
   - Larger triplet margin (2.0) for better separation
   - Cosine distance (better for text similarity)
   - Larger batch size (32) for better learning

3. ANOMALY DETECTION TESTING:
   - Tests if corrupted text has low similarity to clean text
   - Provides detection rate percentage

This ensures corrupted data like "re d" is flagged as anomalous,
while clean data like "blue" remains similar to other clean colors.
"""

import pandas as pd
import numpy as np
import argparse
import json
import re
import random
import os
import sys
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import itertools
from typing import List, Dict, Tuple

# Add the parent directory to the path to import the error injection module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from error_injection import apply_error_rule, load_error_rules

# --- Functions (Error Injection, Data Prep, Training) remain the same ---

def create_semantic_groups(texts: List[str], column_name: str) -> Dict[str, List[str]]:
    """
    Create semantic groups for better negative sampling.
    """
    if column_name == 'article_structure_name_2':
        # Group clothing items by type
        clothing_groups = {
            'outerwear': ['Jackets', 'Coats', 'Blazers'],
            'tops': ['Sweaters', 'Blouses', 'Shirts', 'T-shirts', 'Cardigan'],
            'bottoms': ['Trousers', 'Pants', 'Jeans', 'Skirts'],
            'dresses': ['Dresses'],
            'accessories': ['Bags', 'Scarves', 'Belts']
        }
        
        groups = {}
        for group_name, items in clothing_groups.items():
            groups[group_name] = [text for text in texts if any(item.lower() in text.lower() for item in items)]
        
        # Add remaining items to 'other' group
        assigned_texts = set(itertools.chain.from_iterable(groups.values()))
        remaining = [text for text in texts if text not in assigned_texts]
        if remaining:
            groups['other'] = remaining
            
    elif column_name == 'colour_name':
        # Group colors by families
        color_groups = {
            'neutral': ['White', 'Black', 'Gray', 'Grey', 'Beige', 'Cream', 'Ivory', 'Off White'],
            'warm': ['Red', 'Orange', 'Yellow', 'Pink', 'Coral', 'Peach', 'Warm', 'Camel', 'Mustard'],
            'cool': ['Blue', 'Green', 'Purple', 'Turquoise', 'Teal', 'Navy', 'Cool'],
            'earth': ['Brown', 'Tan', 'Sand', 'Khaki', 'Olive', 'Rust']
        }
        
        groups = {}
        for group_name, colors in color_groups.items():
            groups[group_name] = [text for text in texts if any(color.lower() in text.lower() for color in colors)]
        
        # Add remaining colors to 'other' group
        assigned_texts = set(itertools.chain.from_iterable(groups.values()))
        remaining = [text for text in texts if text not in assigned_texts]
        if remaining:
            groups['other'] = remaining
            
    elif column_name == 'season':
        # For seasons, use TF-IDF clustering since we have limited data
        if len(texts) > 3:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            n_clusters = min(3, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            groups = {}
            for i in range(n_clusters):
                groups[f'cluster_{i}'] = [texts[j] for j in range(len(texts)) if cluster_labels[j] == i]
        else:
            groups = {'all': texts}
            
    elif column_name == 'Care Instructions':
        # Group by care instruction type
        care_groups = {
            'wash_30': [],
            'wash_40': [],
            'no_iron': [],
            'iron_low': [],
            'no_dry': [],
            'dry_clean': []
        }
        
        for text in texts:
            text_lower = text.lower()
            if '30°c' in text_lower or '30°' in text_lower:
                care_groups['wash_30'].append(text)
            elif '40°c' in text_lower or '40°' in text_lower:
                care_groups['wash_40'].append(text)
            
            if 'niet strijken' in text_lower or 'no iron' in text_lower:
                care_groups['no_iron'].append(text)
            elif 'lage temperatuur' in text_lower or 'low temp' in text_lower:
                care_groups['iron_low'].append(text)
            
            if 'niet in de droger' in text_lower or 'no dry' in text_lower:
                care_groups['no_dry'].append(text)
            elif 'dry clean' in text_lower or 'chemisch' in text_lower:
                care_groups['dry_clean'].append(text)
        
        # Remove empty groups
        groups = {k: v for k, v in care_groups.items() if v}
        
        # If no groups found, use all texts
        if not groups:
            groups = {'all': texts}
    else:
        # Default: single group
        groups = {'all': texts}
    
    return groups

def create_improved_triplet_dataset(data_series, rules, column_name):
    """
    Create triplets for anomaly detection - corrupted data should be distant from clean data.
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
    
    # Create semantic groups for better positive sampling
    groups = create_semantic_groups(clean_texts, column_name)
    print(f"Created {len(groups)} semantic groups: {list(groups.keys())}")
    
    # Create reverse mapping: text -> group
    text_to_group = {}
    for group_name, group_texts in groups.items():
        for text in group_texts:
            text_to_group[text] = group_name
    
    triplets = []
    
    for anchor_text in clean_texts:
        # FLIPPED LOGIC FOR ANOMALY DETECTION:
        # Anchor: Clean text (e.g., "red")
        # Positive: Other clean text from same semantic group (e.g., "blue" for colors)
        # Negative: Corrupted version (e.g., "re d") - this should be far from anchor
        
        anchor_group = text_to_group.get(anchor_text, 'all')
        
        # Get positive candidates from same semantic group
        positive_candidates = []
        for group_name, group_texts in groups.items():
            if group_name == anchor_group:
                positive_candidates.extend([text for text in group_texts if text != anchor_text])
        
        # If no candidates from same group, use any other clean text
        if not positive_candidates:
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
    print(f"Structure: Anchor (clean) -> Positive (clean, similar) -> Negative (corrupted)")
    return triplets

def random_hyperparameter_search(df, column, rules, device, num_trials=15):
    """
    Perform thorough random hyperparameter search to find optimal parameters for anomaly detection.
    """
    print(f"\n🔍 Starting THOROUGH hyperparameter search for '{column}' with {num_trials} trials...")
    
    # Define expanded hyperparameter search space for more thorough exploration
    hyperparams_space = {
        'model_name': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        ],
        # Expanded margin range for better separation testing
        'triplet_margin': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
        # All distance metrics for comprehensive testing
        'distance_metric': [
            losses.TripletDistanceMetric.COSINE,
            losses.TripletDistanceMetric.EUCLIDEAN,
            losses.TripletDistanceMetric.MANHATTAN
        ],
        # More batch size options for optimal training
        'batch_size': [8, 16, 24, 32, 48, 64, 96, 128],
        # More epoch options for training duration
        'epochs': [1, 2, 3, 4, 5, 6],
        # Expanded learning rate range
        'learning_rate': [5e-6, 1e-5, 1.5e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4]
    }
    
    best_score = -1
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
        for param_name, values in hyperparams_space.items():
            params[param_name] = random.choice(values)
        
        print(f"Testing parameters: {params}")
        
        try:
            # Train with current parameters
            score = train_with_params(df, column, rules, params)
            results.append((params.copy(), score))
            
            # Track performance by parameter
            model_name = params['model_name'].split('/')[-1]
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(score)
            
            margin = params['triplet_margin']
            if margin not in margin_scores:
                margin_scores[margin] = []
            margin_scores[margin].append(score)
            
            distance = str(params['distance_metric']).split('.')[-1]
            if distance not in distance_scores:
                distance_scores[distance] = []
            distance_scores[distance].append(score)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"🎉 New best score: {score:.4f}")
                
        except Exception as e:
            print(f"❌ Trial {trial + 1} failed: {e}")
            results.append((params.copy(), -1))
    
    # Print analysis of parameter performance
    print(f"\n📊 Parameter Performance Analysis:")
    print(f"{'='*50}")
    
    print("\n🤖 Model Performance:")
    for model, scores in model_scores.items():
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {model}: {avg_score:.4f} (avg from {len(scores)} trials)")
    
    print("\n📏 Triplet Margin Performance:")
    for margin, scores in sorted(margin_scores.items()):
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {margin}: {avg_score:.4f} (avg from {len(scores)} trials)")
    
    print("\n📐 Distance Metric Performance:")
    for distance, scores in distance_scores.items():
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {distance}: {avg_score:.4f} (avg from {len(scores)} trials)")
    
    print(f"\n🏆 Best parameters found for anomaly detection:")
    print(f"Score: {best_score:.4f}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params, best_score, results

def train_with_params(df, column, rules, params):
    """
    Train model with specific hyperparameters and return validation score.
    """
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    if not triplets:
        return -1
    
    # Initialize model
    model = SentenceTransformer(params['model_name'])
    
    # Split data
    split_idx = int(len(triplets) * 0.8)
    train_triplets = triplets[:split_idx]
    eval_triplets = triplets[split_idx:]
    
    if not eval_triplets:
        return -1
    
    # Create loss function with hyperparameters
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=params['distance_metric'],
        triplet_margin=params['triplet_margin']
    )
    
    # Create evaluator
    evaluator = TripletEvaluator.from_input_examples(
        eval_triplets, 
        name=f'{column}_hp_search'
    )
    
    # Train model
    model.fit(
        train_objectives=[(DataLoader(train_triplets, shuffle=True, batch_size=params['batch_size']), train_loss)],
        epochs=params['epochs'],
        evaluator=evaluator,
        evaluation_steps=max(1, len(train_triplets) // params['batch_size']),
        warmup_steps=50,
        output_path=f'./hp_search_{column.replace(" ", "_").lower()}',
        save_best_model=False,
        show_progress_bar=False
    )
    
    # Get final score
    final_results = evaluator(model)
    if isinstance(final_results, dict):
        for key, value in final_results.items():
            if 'accuracy' in key.lower():
                return value
    
    return final_results if isinstance(final_results, (int, float)) else -1

def get_optimal_parameters(column_name, fallback_model_name, fallback_epochs):
    """
    Get optimal parameters for each column based on hyperparameter search results.
    """
    # Optimal parameters discovered from hyperparameter search
    optimal_params = {
        'article_structure_name_2': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 1.0,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 48,
            'epochs': 2,
            'learning_rate': 2e-5
        },
        'colour_name': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'triplet_margin': 2.5,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 128,
            'epochs': 1,
            'learning_rate': 5e-6
        },
        # Default parameters for other columns
        'season': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'triplet_margin': 1.5,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 32,
            'epochs': 2,
            'learning_rate': 2e-5
        },
        'Care Instructions': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'triplet_margin': 1.5,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 32,
            'epochs': 2,
            'learning_rate': 2e-5
        }
    }
    
    if column_name in optimal_params:
        return optimal_params[column_name]
    else:
        # Fallback to general optimal parameters
        return {
            'model_name': fallback_model_name,
            'triplet_margin': 1.5,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 32,
            'epochs': fallback_epochs,
            'learning_rate': 2e-5
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

def train_and_evaluate_similarity_model(df, column, rules, device, model_name, num_epochs, use_hp_search=False):
    """
    Train a sentence transformer model for anomaly detection using triplet loss.
    """
    if use_hp_search:
        best_params, best_score, search_results = random_hyperparameter_search(df, column, rules, device, num_trials=args.hp_trials)
        if best_score <= 0:
            print(f"Hyperparameter search failed for '{column}'. Using optimal parameters.")
            best_params = get_optimal_parameters(column, model_name, num_epochs)
    else:
        # Use optimal parameters based on hyperparameter search results
        best_params = get_optimal_parameters(column, model_name, num_epochs)
        print(f"Using pre-optimized parameters for '{column}' (based on hyperparameter search results)")
    
    # Train final model with best parameters
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    if not triplets:
        print(f"Could not create triplets for '{column}'. Skipping.")
        return None
    
    print(f"Training final model with optimized parameters for anomaly detection...")
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
        evaluator = TripletEvaluator.from_input_examples(eval_triplets, name=f'{column}_final_evaluation')
        evaluation_steps = max(1, len(train_triplets) // best_params['batch_size'])
    else:
        evaluator = None
        evaluation_steps = 0
    
    # Train the model
    print(f"Training anomaly detection model for '{column}' with {len(train_triplets)} triplets...")
    model.fit(
        train_objectives=[(DataLoader(train_triplets, shuffle=True, batch_size=best_params['batch_size']), train_loss)],
        epochs=best_params['epochs'],
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        warmup_steps=50,
        output_path=f'./results_{column.replace(" ", "_").lower()}',
        save_best_model=True,
        show_progress_bar=True
    )
    
    # Evaluate final performance
    if evaluator:
        print(f"\n--- Final Performance for column: '{column}' ---")
        final_results = evaluator(model, output_path=f'./results_{column.replace(" ", "_").lower()}')
        if isinstance(final_results, dict):
            # Extract the accuracy score from the results dictionary
            for key, value in final_results.items():
                if 'accuracy' in key.lower():
                    print(f"  - {key}: {value:.4f}")
                    break
        else:
            print(f"  - Triplet Accuracy: {final_results:.4f}")
    
    # Test anomaly detection capability
    clean_texts = df[column].dropna().apply(preprocess_text).astype(str).tolist()
    anomaly_detection_rate = test_anomaly_detection(model, clean_texts, rules, column)
    
    # Demonstrate anomaly detection with examples
    demonstrate_similarity(model, df[column], column)
    
    return model

def demonstrate_similarity(model, data_series, column_name):
    """
    Demonstrate how the model calculates similarity between texts for anomaly detection
    """
    print(f"\n--- Anomaly Detection Demonstration for '{column_name}' ---")
    
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
    
    # Load rules to create corrupted examples
    rules_dir = '../../rules'
    rule_files = {
        'article_structure_name_2': 'category.json',
        'colour_name': 'color_name.json',
        'season': 'season.json',
        'Care Instructions': 'care_instructions.json'
    }
    
    rule_file = rule_files.get(column_name, 'category.json')
    rule_path = os.path.join(rules_dir, rule_file)
    
    rules = []
    try:
        rules = load_error_rules(rule_path)
    except FileNotFoundError:
        print(f"Rule file '{rule_path}' not found. Creating simple corruption examples.")
        rules = [
            {'operation': 'add_whitespace'},
            {'operation': 'random_noise'},
            {'operation': 'string_replace', 'params': {'find': 'e', 'replace': 'E'}}
        ]
    
    if rules and len(sample_texts) > 0:
        # Take first 3 clean samples
        test_samples = sample_texts[:3]
        
        print("Testing anomaly detection (lower similarity = better anomaly detection):")
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
                
                print(f"  {i+1}. Clean: '{clean_text}' vs Corrupted: '{corrupted_text}'")
                print(f"     Similarity: {similarity:.4f} {'✅ GOOD (low similarity = anomaly detected)' if similarity < 0.8 else '❌ BAD (high similarity = anomaly missed)'}")
    
    # Show embeddings dimensionality
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Total samples processed: {len(sample_texts)}")
    print(f"💡 For anomaly detection: Clean texts should be similar to each other, corrupted texts should be dissimilar.")

def test_anomaly_detection(model, clean_texts, rules, column_name, threshold=0.7):
    """
    Test the trained model's ability to detect anomalies.
    Returns the percentage of anomalies correctly detected.
    """
    print(f"\n🔍 Testing anomaly detection for '{column_name}'...")
    
    if not rules or len(clean_texts) < 2:
        print("Not enough data to test anomaly detection.")
        return 0.0
    
    # Preprocess clean texts
    clean_texts = [preprocess_text(text) for text in clean_texts]
    
    # Create test set
    test_clean = clean_texts[:min(20, len(clean_texts))]  # Limit to 20 for speed
    detected_anomalies = 0
    total_tests = 0
    
    print(f"Testing with {len(test_clean)} clean samples and threshold {threshold}")
    
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
        print(f"\n🎯 Anomaly Detection Rate: {detection_rate:.1f}% ({detected_anomalies}/{total_tests})")
        return detection_rate
    else:
        print("No anomalies could be generated for testing.")
        return 0.0


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune similarity models for anomaly detection.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    parser.add_argument("--use-hp-search", action="store_true", help="Use hyperparameter search to find optimal parameters.")
    parser.add_argument("--hp-trials", type=int, default=15, help="Number of hyperparameter search trials (default: 15 for thorough search).")
    args = parser.parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps"); print("✅ Apple M1/M2 GPU found. Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("✅ NVIDIA GPU found. Using CUDA.")
    else:
        device = torch.device("cpu"); print("⚠️ No GPU found. Using CPU.")

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}"); exit()
        
    rule_to_column_map = {
        "category": "article_structure_name_2",
        "color_name": "colour_name",
        "ean": "EAN",
        "article_number": "article_number",
        "colour_code": "colour_code",
        "customs_tariff_number": "customs_tariff_number",
        "description_short_1": "description_short_1",
        "long_description_nl": "long_description_NL",
        "material": "material",
        "product_name_en": "product_name_EN",
        "size_name": "size_name",
        # Excluded: season (only 1 unique value), care_instructions (only 2 unique values)
    }

    column_configs = {
        # Updated with optimal parameters from hyperparameter search
        'Care Instructions':        {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'colour_name':              {'model': 'sentence-transformers/all-mpnet-base-v2', 'epochs': 1},  # Optimal from search
        'season':                   {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'article_structure_name_2': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 2},  # Optimal from search
        # New columns with default configurations
        'EAN':                      {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'article_number':           {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'colour_code':              {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'customs_tariff_number':    {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'description_short_1':      {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'long_description_NL':      {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'material':                 {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'product_name_EN':          {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'size_name':                {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
    }

    rules_dir = '../../rules'
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    for rule_name, column_name in rule_to_column_map.items():
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in the CSV. Skipping.")
            continue
        
        config = column_configs.get(column_name, {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2})
        
        print(f"\n{'='*20} Starting Process for Column: {column_name} {'='*20}")
        print(f"Using rule file: '{rule_name}.json', Model: {config['model']}, Epochs: {config['epochs']}")
        
        if args.use_hp_search:
            print(f"🔍 Hyperparameter search enabled with {args.hp_trials} trials")
        
        file_path = os.path.join(rules_dir, f'{rule_name}.json')
        rules = []
        try:
            rules = load_error_rules(file_path)
        except FileNotFoundError:
            print(f"Error: Rule file '{file_path}' not found.")
            continue
        
        if not rules:
            print(f"No rules found in '{file_path}'. Skipping.")
            continue
            
        train_and_evaluate_similarity_model(
            df, 
            column_name, 
            rules, 
            device=device, 
            model_name=config['model'],
            num_epochs=config['epochs'],
            use_hp_search=args.use_hp_search
        )