"""
RECALL-FOCUSED Anomaly Detection using Sentence Transformers and Triplet Loss

📊 ACTUAL PERFORMANCE RESULTS FROM 5-TRIAL HYPERPARAMETER SEARCH:

EXCELLENT PERFORMANCE (1.0 recall):
- article_structure_name_2: 1.0 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 0.3, epochs: 4) ✅
- colour_code: 1.0 recall → all-mpnet-base-v2 (margin: 2.0, epochs: 2) ✅
- colour_name: 1.0 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 0.3, epochs: 4) ✅

GOOD PERFORMANCE (0.8+ recall):
- material: 0.882 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 1.0, epochs: 3) 🔥

MODERATE PERFORMANCE (0.4-0.5 recall):
- EAN: 0.5 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 0.3, epochs: 3) ⚡
- long_description_NL: 0.5 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 1.0, epochs: 3) ⚡
- customs_tariff_number: 0.5 recall → multi-qa-MiniLM-L6-cos-v1 (margin: 1.2, epochs: 2) ⚡
- size_name: 0.4 recall → paraphrase-MiniLM-L6-v2 (margin: 0.3, epochs: 8) ⚡

POOR PERFORMANCE (0.0 recall):
- article_number: 0.0 recall → all-mpnet-base-v2 (margin: 1.0, epochs: 5) ❌
- description_short_1: 0.0 recall → distilbert-base-nli-stsb-mean-tokens (margin: 2.0, epochs: 5) ❌
- product_name_EN: 0.0 recall → distilbert-base-nli-stsb-mean-tokens (margin: 0.8, epochs: 6) ❌

📁 ORGANIZED OUTPUT STRUCTURE:
../results/
├── summary/                    (HP search results, summaries)
│   ├── hp_search_results_*.json
│   └── hp_search_summary.json
├── checkpoints/                (temporary training checkpoints)
└── results_[column]/           (trained models for each column)
    ├── model files
    ├── config files
    └── evaluation results

Key Insights from ACTUAL Hyperparameter Search:
1. DIFFERENT MODELS WORK BETTER FOR DIFFERENT COLUMNS:
   - multi-qa-MiniLM-L6-cos-v1: BEST overall - excellent for structured data (article_structure_name_2, colour_name, material, EAN, long_description_NL, customs_tariff_number)
   - all-mpnet-base-v2: Good for color codes and article numbers
   - paraphrase-MiniLM-L6-v2: Best for size names
   - distilbert-base-nli-stsb-mean-tokens: Works for product names and descriptions but with poor recall

2. RECALL-FOCUSED CONFIGURATION INSIGHTS:
   - Small margins (0.3-1.0) work best for most columns
   - COSINE distance metric is most effective for text similarity
   - Lower epochs (2-4) often sufficient for good performance
   - Batch sizes 16-64 work well depending on column complexity

3. PERFORMANCE PATTERNS:
   - Structured data (article_structure_name_2, colour_name): Perfect recall (1.0)
   - Material data: Very good recall (0.882)
   - Identifier data (EAN, customs_tariff_number): Moderate recall (0.5)
   - Descriptive text (product_name_EN, description_short_1): Poor recall (0.0)

4. FLIPPED TRIPLET LOGIC:
   - Anchor: Clean text (e.g., "red")
   - Positive: Other clean text from same semantic group (e.g., "blue")
   - Negative: Corrupted text (e.g., "re d") - should be far from anchor

This prioritizes catching anomalies over precision - better to flag clean data
as anomalous than to miss actual anomalies.
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
    Perform RECALL-FOCUSED hyperparameter search for anomaly detection.
    """
    print(f"\n🎯 Starting RECALL-FOCUSED hyperparameter search for '{column}' with {num_trials} trials...")
    
    # Define RECALL-OPTIMIZED hyperparameter search space
    hyperparams_space = {
        'model_name': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        ],
        # SMALLER margins for recall optimization (less strict separation)
        'triplet_margin': [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        # Focus on cosine for text similarity
        'distance_metric': [
            losses.TripletDistanceMetric.COSINE,
            losses.TripletDistanceMetric.EUCLIDEAN,
            losses.TripletDistanceMetric.MANHATTAN
        ],
        # Smaller batch sizes for better recall learning
        'batch_size': [8, 16, 24, 32, 48, 64],
        # MORE epochs for recall optimization
        'epochs': [2, 3, 4, 5, 6, 8],
        # Lower learning rates for stable training
        'learning_rate': [1e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
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
            # Train with current parameters and get RECALL score
            recall_score = train_with_params(df, column, rules, params)
            results.append((params.copy(), recall_score))
            
            # Track performance by parameter
            model_name = params['model_name'].split('/')[-1]
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(recall_score)
            
            margin = params['triplet_margin']
            if margin not in margin_scores:
                margin_scores[margin] = []
            margin_scores[margin].append(recall_score)
            
            distance = str(params['distance_metric']).split('.')[-1]
            if distance not in distance_scores:
                distance_scores[distance] = []
            distance_scores[distance].append(recall_score)
            
            if recall_score > best_score:
                best_score = recall_score
                best_params = params.copy()
                print(f"� New best RECALL score: {recall_score:.4f} ({recall_score*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Trial {trial + 1} failed: {e}")
            results.append((params.copy(), -1))
    
    # Print analysis of parameter performance
    print(f"\n📊 RECALL-FOCUSED Parameter Performance Analysis:")
    print(f"{'='*60}")
    
    print("\n🤖 Model Performance (RECALL Score):")
    for model, scores in model_scores.items():
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {model}: {avg_score:.4f} ({avg_score*100:.1f}% recall)")
    
    print("\n📏 Triplet Margin Performance (RECALL Score):")
    for margin, scores in sorted(margin_scores.items()):
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {margin}: {avg_score:.4f} ({avg_score*100:.1f}% recall)")
    
    print("\n📐 Distance Metric Performance (RECALL Score):")
    for distance, scores in distance_scores.items():
        avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        print(f"  {distance}: {avg_score:.4f} ({avg_score*100:.1f}% recall)")
    
    print(f"\n🏆 Best RECALL-OPTIMIZED parameters:")
    print(f"RECALL Score: {best_score:.4f} ({best_score*100:.1f}%)")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save hyperparameter search results to file
    results_dir = os.path.join('..', 'results', 'summary')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"hp_search_results_{column.replace(' ', '_').lower()}.json")
    hp_results = {
        'column': column,
        'best_params': best_params,
        'best_score': best_score,
        'all_results': [(params, score) for params, score in results],
        'performance_analysis': {
            'model_scores': {model: sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0 
                           for model, scores in model_scores.items()},
            'margin_scores': {str(margin): sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0 
                            for margin, scores in margin_scores.items()},
            'distance_scores': {distance: sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0 
                              for distance, scores in distance_scores.items()}
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(hp_results, f, indent=2, default=str)
    
    print(f"📁 Hyperparameter search results saved to: {results_file}")
    
    return best_params, best_score, results

def train_with_params(df, column, rules, params):
    """
    Train model with specific hyperparameters and return RECALL score.
    """
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    if not triplets:
        return -1
    
    # Get clean texts for recall evaluation
    clean_texts = df[column].dropna().apply(preprocess_text).astype(str).tolist()
    if len(clean_texts) < 5:
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
    
    # Ensure checkpoints directory exists for temporary training during HP search
    checkpoints_dir = os.path.join('..', 'results', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Train model WITHOUT evaluator and WITHOUT saving (we'll evaluate recall separately)
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
    
    # Evaluate RECALL performance instead of triplet accuracy
    try:
        recall_score = evaluate_recall_performance(model, clean_texts, rules, column)
        return recall_score
    except Exception as e:
        print(f"   ❌ Recall evaluation failed: {e}")
        return -1


def evaluate_recall_performance(model, clean_texts, rules, column_name, num_samples=20):
    """
    Evaluate model's recall performance on anomaly detection.
    Returns recall score (0-1).
    """
    if not rules or len(clean_texts) < 2:
        return 0.0
    
    # Take sample of clean texts for evaluation
    test_clean = clean_texts[:min(num_samples, len(clean_texts))]
    detected_anomalies = 0
    total_tests = 0
    
    # Use recall-optimized threshold
    threshold = 0.6
    
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
            
            total_tests += 1
    
    # Return recall score (0-1)
    if total_tests > 0:
        recall = detected_anomalies / total_tests
        return recall
    else:
        return 0.0

def get_optimal_parameters(column_name, fallback_model_name, fallback_epochs):
    """
    Get RECALL-OPTIMIZED parameters for each column based on ACTUAL hyperparameter search results.
    Updated with real performance data from 5-trial hyperparameter search.
    """
    # ACTUAL BEST PARAMETERS from hyperparameter search results
    optimal_params = {
        # EXCELLENT PERFORMANCE (1.0 recall) - Proven successful configurations
        'article_structure_name_2': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 0.3,  # ACTUAL best found
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 64,  # ACTUAL best found
            'epochs': 4,  # ACTUAL best found
            'learning_rate': 5e-06  # ACTUAL best found
        },
        'colour_code': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'triplet_margin': 2.0,  # ACTUAL best found
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 16,  # ACTUAL best found
            'epochs': 2,  # ACTUAL best found
            'learning_rate': 1e-06  # ACTUAL best found
        },
        'colour_name': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 0.3,  # ACTUAL best found (score: 1.0)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 64,  # ACTUAL best found
            'epochs': 4,  # ACTUAL best found
            'learning_rate': 5e-06  # ACTUAL best found
        },
        
        # GOOD PERFORMANCE (0.8+ recall) - Use actual best found
        'material': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 1.0,  # ACTUAL best found (score: 0.882)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 48,  # ACTUAL best found
            'epochs': 3,  # ACTUAL best found
            'learning_rate': 5e-06  # ACTUAL best found
        },
        
        # MODERATE PERFORMANCE (0.4-0.5 recall) - Use actual best found
        'EAN': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 0.3,  # ACTUAL best found (score: 0.5)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 16,  # ACTUAL best found
            'epochs': 3,  # ACTUAL best found
            'learning_rate': 1e-06  # ACTUAL best found
        },
        'long_description_NL': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 1.0,  # ACTUAL best found (score: 0.5)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 48,  # ACTUAL best found
            'epochs': 3,  # ACTUAL best found
            'learning_rate': 5e-06  # ACTUAL best found
        },
        'customs_tariff_number': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'triplet_margin': 1.2,  # ACTUAL best found (score: 0.5)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 48,  # ACTUAL best found
            'epochs': 2,  # ACTUAL best found
            'learning_rate': 1e-05  # ACTUAL best found
        },
        'size_name': {
            'model_name': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'triplet_margin': 0.3,  # ACTUAL best found (score: 0.4)
            'distance_metric': losses.TripletDistanceMetric.MANHATTAN,  # ACTUAL best found
            'batch_size': 48,  # ACTUAL best found
            'epochs': 8,  # ACTUAL best found
            'learning_rate': 2e-05  # ACTUAL best found
        },
        
        # POOR PERFORMANCE (0.0 recall) - Use actual best found, but may need further work
        'article_number': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'triplet_margin': 1.0,  # ACTUAL best found (score: 0.0)
            'distance_metric': losses.TripletDistanceMetric.EUCLIDEAN,  # Using lambda function reference
            'batch_size': 16,  # ACTUAL best found
            'epochs': 5,  # ACTUAL best found
            'learning_rate': 5e-06  # ACTUAL best found
        },
        'description_short_1': {
            'model_name': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'triplet_margin': 2.0,  # ACTUAL best found (score: 0.0)
            'distance_metric': losses.TripletDistanceMetric.MANHATTAN,  # Using lambda function reference
            'batch_size': 24,  # ACTUAL best found
            'epochs': 5,  # ACTUAL best found
            'learning_rate': 1e-05  # ACTUAL best found
        },
        'product_name_EN': {
            'model_name': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'triplet_margin': 0.8,  # ACTUAL best found (score: 0.0)
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 32,  # ACTUAL best found
            'epochs': 6,  # ACTUAL best found
            'learning_rate': 1e-06  # ACTUAL best found
        },
        
        # DEFAULT CONFIGURATIONS (not tested yet)
        'season': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'triplet_margin': 0.8,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 24,
            'epochs': 4,
            'learning_rate': 1e-5
        },
        'Care Instructions': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'triplet_margin': 0.8,
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 24,
            'epochs': 4,
            'learning_rate': 1e-5
        }
    }
    
    if column_name in optimal_params:
        return optimal_params[column_name]
    else:
        # Fallback to recall-optimized parameters
        return {
            'model_name': fallback_model_name,
            'triplet_margin': 0.8,  # Smaller for better recall
            'distance_metric': losses.TripletDistanceMetric.COSINE,
            'batch_size': 24,
            'epochs': max(3, fallback_epochs),  # At least 3 epochs for recall
            'learning_rate': 1e-5
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
    Train a sentence transformer model for RECALL-FOCUSED anomaly detection using triplet loss.
    """
    print(f"\n🎯 Training RECALL-OPTIMIZED model for '{column}'")
    
    # Create results directory structure
    results_base_dir = os.path.join('..', 'results')
    model_results_dir = os.path.join(results_base_dir, f'results_{column.replace(" ", "_").lower()}')
    checkpoints_dir = os.path.join(results_base_dir, 'checkpoints')
    
    # Ensure all directories exist
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    if use_hp_search:
        best_params, best_score, search_results = random_hyperparameter_search(df, column, rules, device, num_trials=args.hp_trials)
        if best_score <= 0:
            print(f"Hyperparameter search failed for '{column}'. Using recall-optimized parameters.")
            best_params = get_optimal_parameters(column, model_name, num_epochs)
    else:
        # Use recall-optimized parameters
        best_params = get_optimal_parameters(column, model_name, num_epochs)
        print(f"Using RECALL-OPTIMIZED parameters for '{column}'")
    
    # Train final model with best parameters
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    if not triplets:
        print(f"Could not create triplets for '{column}'. Skipping.")
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
        evaluator = TripletEvaluator.from_input_examples(eval_triplets, name=f'{column}_final_evaluation')
        evaluation_steps = max(1, len(train_triplets) // best_params['batch_size'])
    else:
        evaluator = None
        evaluation_steps = 0
    
    # Train the model with organized output structure
    print(f"Training RECALL-OPTIMIZED anomaly detection model for '{column}' with {len(train_triplets)} triplets...")
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
        print(f"\n--- RECALL-FOCUSED Performance for column: '{column}' ---")
        final_results = evaluator(model, output_path=model_results_dir)
        if isinstance(final_results, dict):
            # Extract the accuracy score from the results dictionary
            for key, value in final_results.items():
                if 'accuracy' in key.lower():
                    print(f"  - {key}: {value:.4f}")
                    break
        else:
            print(f"  - Triplet Accuracy: {final_results:.4f}")
    
    # Test RECALL-FOCUSED anomaly detection capability
    clean_texts = df[column].dropna().apply(preprocess_text).astype(str).tolist()
    print(f"\n💡 Testing with RECALL-OPTIMIZED threshold (lower = more sensitive)")
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
            {'operation': 'random_noise'},
            {'operation': 'string_replace', 'params': {'find': 'e', 'replace': 'E'}}
        ]
    
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
                
                print(f"  {i+1}. Clean: '{clean_text}' vs Corrupted: '{corrupted_text}'")
                # Use recall-focused threshold (0.6 instead of 0.8)
                print(f"     Similarity: {similarity:.4f} {'✅ DETECTED (good recall)' if similarity < 0.6 else '⚠️ MISSED (need lower threshold)'}")
    
    # Show embeddings dimensionality
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Total samples processed: {len(sample_texts)}")
    print(f"🎯 RECALL-FOCUSED: Lower thresholds catch more anomalies (better recall, may increase false positives)")

def test_anomaly_detection(model, clean_texts, rules, column_name, threshold=0.6):
    """
    Test the trained model's ability to detect anomalies with RECALL FOCUS.
    Uses lower threshold (0.6) to catch more anomalies.
    """
    print(f"\n🎯 Testing RECALL-FOCUSED anomaly detection for '{column_name}'...")
    
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


def save_aggregated_hp_results():
    """
    Read all individual hyperparameter search results and create an aggregated summary.
    """
    import glob
    
    # Look for HP search results in the results/summary directory
    results_dir = os.path.join('..', 'results', 'summary')
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
                column = data['column']
                aggregated_results[column] = {
                    'best_score': data['best_score'],
                    'best_params': data['best_params'],
                    'performance_analysis': data['performance_analysis']
                }
                all_best_params[column] = data['best_params']
        except Exception as e:
            print(f"Error reading {hp_file}: {e}")
    
    # Save aggregated results in the summary folder
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_columns': len(aggregated_results),
        'column_results': aggregated_results,
        'recommended_configs': all_best_params
    }
    
    summary_file = os.path.join(results_dir, 'hp_search_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📋 HYPERPARAMETER SEARCH SUMMARY:")
    print(f"{'='*60}")
    print(f"Total columns tested: {len(aggregated_results)}")
    print(f"\nRanked by RECALL Score:")
    
    # Sort by recall score
    sorted_results = sorted(aggregated_results.items(), key=lambda x: x[1]['best_score'], reverse=True)
    
    for column, result in sorted_results:
        score = result['best_score']
        model = result['best_params']['model_name'].split('/')[-1]
        margin = result['best_params']['triplet_margin']
        epochs = result['best_params']['epochs']
        batch_size = result['best_params']['batch_size']
        
        print(f"  {column}: {score:.4f} ({score*100:.1f}%)")
        print(f"    Model: {model}")
        print(f"    Margin: {margin}, Epochs: {epochs}, Batch: {batch_size}")
    
    print(f"\n📁 Full summary saved to: {summary_file}")
    return summary


def setup_results_directory_structure():
    """
    Create the organized directory structure for all results and outputs.
    """
    base_results_dir = os.path.join('..', 'results')
    
    # Create main directories
    directories = [
        base_results_dir,
        os.path.join(base_results_dir, 'summary'),          # For HP search results and summaries
        os.path.join(base_results_dir, 'checkpoints'),      # For temporary training checkpoints
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"📁 Results directory structure created:")
    print(f"  - {base_results_dir}/")
    print(f"    ├── summary/          (HP search results, summaries)")
    print(f"    ├── checkpoints/      (temporary training checkpoints)")
    print(f"    └── results_[column]/ (trained models for each column)")
    
    return base_results_dir


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECALL-FOCUSED anomaly detection using sentence transformers.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    parser.add_argument("--use-hp-search", action="store_true", help="Use RECALL-FOCUSED hyperparameter search.")
    parser.add_argument("--hp-trials", type=int, default=15, help="Number of hyperparameter search trials (default: 15).")
    args = parser.parse_args()
    
    print("🎯 RECALL-OPTIMIZED Anomaly Detection Training")
    print("💡 Strategy: Better to flag clean data as anomalous than to miss actual anomalies")
    
    # Setup organized directory structure for all outputs
    setup_results_directory_structure()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps"); print("Apple M1/M2 GPU found. Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("NVIDIA GPU found. Using CUDA.")
    else:
        device = torch.device("cpu"); print("No GPU found. Using CPU.")

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}"); exit()
        
    rule_to_column_map = {
        # "category": "article_structure_name_2",
        "color_name": "colour_name",
        # "ean": "EAN",
        "article_number": "article_number",
        # "colour_code": "colour_code",
        "customs_tariff_number": "customs_tariff_number",
        "description_short_1": "description_short_1",
        # "long_description_nl": "long_description_NL",
        "material": "material",
        "product_name_en": "product_name_EN",
        # "size": "size_name",  # Fixed: use 'size' rule file for 'size_name' column
        # Excluded: season (only 1 unique value), care_instructions (only 2 unique values)
    }

    column_configs = {
        # EXCELLENT PERFORMANCE (1.0 recall) - Use ACTUAL best found parameters
        'article_structure_name_2': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 4},  # 1.0 recall
        'colour_code': {'model': 'sentence-transformers/all-mpnet-base-v2', 'epochs': 2},  # 1.0 recall
        'colour_name': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 4},  # 1.0 recall
        
        # GOOD PERFORMANCE (0.8+ recall) - Use ACTUAL best found parameters
        'material': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.882 recall
        
        # MODERATE PERFORMANCE (0.4-0.5 recall) - Use ACTUAL best found parameters
        'EAN': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.5 recall
        'long_description_NL': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 3},  # 0.5 recall
        'customs_tariff_number': {'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'epochs': 2},  # 0.5 recall
        'size_name': {'model': 'sentence-transformers/paraphrase-MiniLM-L6-v2', 'epochs': 8},  # 0.4 recall
        
        # POOR PERFORMANCE (0.0 recall) - Use ACTUAL best found, but may need further optimization
        'article_number': {'model': 'sentence-transformers/all-mpnet-base-v2', 'epochs': 5},  # 0.0 recall
        'description_short_1': {'model': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens', 'epochs': 5},  # 0.0 recall
        'product_name_EN': {'model': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens', 'epochs': 6},  # 0.0 recall
        
        # DEFAULT CONFIGURATIONS (not tested yet)
        'Care Instructions': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'season': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
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
            print(f"Hyperparameter search enabled with {args.hp_trials} trials")
        
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
    
    # Save aggregated hyperparameter search results if HP search was used
    if args.use_hp_search:
        save_aggregated_hp_results()