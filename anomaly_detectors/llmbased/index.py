import pandas as pd
import numpy as np
import argparse
import json
import re
import random
import os
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

# --- Functions (Error Injection, Data Prep, Training) remain the same ---

def apply_error_rule(value, rule):
    if 'conditions' in rule and rule['conditions']:
        should_apply = False
        for cond in rule['conditions']:
            if cond['type'] == 'contains' and str(cond['value']) in str(value):
                should_apply = True; break
        if not should_apply: return value
    op, params = rule['operation'], rule.get('params', {})
    val_str = str(value)
    if op == 'string_replace': return val_str.replace(str(params['find']), str(params['replace']))
    if op == 'regex_replace': return re.sub(params['pattern'], params['replace'], val_str, count=params.get('count', 0))
    if op == 'add_whitespace': return f" {val_str} "
    if op == 'append': return val_str + params['text']
    if op == 'prepend': return params['text'] + val_str
    if op == 'replace_with': return params['text']
    if op == 'random_noise':
        if not val_str: return val_str
        pos = random.randint(0, len(val_str)); char = random.choice('!@#$%^&*()[]{}|;:",./<>?')
        return val_str[:pos] + char + val_str[pos:]
    return value

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
    Create triplets with improved negative sampling using semantic groups.
    """
    print("Generating improved triplet dataset for similarity learning...")
    clean_texts = data_series.dropna().astype(str).tolist()
    
    if not rules:
        print("Warning: No rules provided. Cannot generate triplet data.")
        return []
    
    if len(clean_texts) < 2:
        print("Warning: Need at least 2 clean texts to create triplets.")
        return []
    
    # Create semantic groups for better negative sampling
    groups = create_semantic_groups(clean_texts, column_name)
    print(f"Created {len(groups)} semantic groups: {list(groups.keys())}")
    
    # Create reverse mapping: text -> group
    text_to_group = {}
    for group_name, group_texts in groups.items():
        for text in group_texts:
            text_to_group[text] = group_name
    
    triplets = []
    
    for anchor_text in clean_texts:
        # Create positive example by applying error rules (similar variant)
        rule = random.choice(rules)
        positive_text = apply_error_rule(anchor_text, rule)
        
        # If rule didn't change the text, try another rule
        if positive_text == anchor_text and len(rules) > 1:
            positive_text = apply_error_rule(anchor_text, random.choice(rules))
        
        # Improved negative sampling: select from different semantic group
        anchor_group = text_to_group.get(anchor_text, 'all')
        
        # Get candidates from different groups
        negative_candidates = []
        for group_name, group_texts in groups.items():
            if group_name != anchor_group:
                negative_candidates.extend(group_texts)
        
        # If no candidates from other groups, use all except anchor
        if not negative_candidates:
            negative_candidates = [text for text in clean_texts if text != anchor_text]
        
        if negative_candidates:
            negative_text = random.choice(negative_candidates)
            
            # Create InputExample for sentence transformers
            triplet = InputExample(
                texts=[anchor_text, positive_text, negative_text],
                label=0  # Not used in triplet loss
            )
            triplets.append(triplet)
    
    print(f"Created {len(triplets)} triplets for training.")
    return triplets

def random_hyperparameter_search(df, column, rules, device, num_trials=5):
    """
    Perform random hyperparameter search to find optimal parameters.
    """
    print(f"\n🔍 Starting hyperparameter search for '{column}' with {num_trials} trials...")
    
    # Define hyperparameter search space
    hyperparams_space = {
        'model_name': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
        ],
        'triplet_margin': [0.5, 1.0, 2.0, 5.0],
        'distance_metric': [
            losses.TripletDistanceMetric.EUCLIDEAN,
            losses.TripletDistanceMetric.COSINE,
            losses.TripletDistanceMetric.MANHATTAN
        ],
        'batch_size': [8, 16, 32],
        'epochs': [1, 2, 3],
        'learning_rate': [1e-5, 2e-5, 5e-5]
    }
    
    best_score = -1
    best_params = None
    results = []
    
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
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"🎉 New best score: {score:.4f}")
            
        except Exception as e:
            print(f"❌ Trial {trial + 1} failed: {e}")
            results.append((params.copy(), -1))
    
    print(f"\n🏆 Best parameters found:")
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

def train_and_evaluate_similarity_model(df, column, rules, device, model_name, num_epochs, use_hp_search=False):
    """
    Train a sentence transformer model for similarity learning using triplet loss.
    """
    if use_hp_search:
        best_params, best_score, search_results = random_hyperparameter_search(df, column, rules, device, num_trials=5)
        if best_score <= 0:
            print(f"Hyperparameter search failed for '{column}'. Using default parameters.")
            best_params = {
                'model_name': model_name,
                'triplet_margin': 5.0,
                'distance_metric': losses.TripletDistanceMetric.EUCLIDEAN,
                'batch_size': 16,
                'epochs': num_epochs,
                'learning_rate': 2e-5
            }
    else:
        best_params = {
            'model_name': model_name,
            'triplet_margin': 5.0,
            'distance_metric': losses.TripletDistanceMetric.EUCLIDEAN,
            'batch_size': 16,
            'epochs': num_epochs,
            'learning_rate': 2e-5
        }
    
    # Train final model with best parameters
    triplets = create_improved_triplet_dataset(df[column], rules, column)
    if not triplets:
        print(f"Could not create triplets for '{column}'. Skipping.")
        return None
    
    print(f"Training final model with optimized parameters...")
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
    print(f"Training similarity model for '{column}' with {len(train_triplets)} triplets...")
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
    
    # Demonstrate similarity with examples
    demonstrate_similarity(model, df[column], column)
    
    return model

def demonstrate_similarity(model, data_series, column_name):
    """
    Demonstrate how the model calculates similarity between texts
    """
    print(f"\n--- Similarity Demonstration for '{column_name}' ---")
    
    # Get some sample texts
    sample_texts = data_series.dropna().astype(str).unique()[:10]
    
    if len(sample_texts) < 2:
        print("Not enough unique texts to demonstrate similarity.")
        return
    
    # Generate embeddings for all samples
    embeddings = model.encode(sample_texts)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Show top 3 most similar pairs
    print("Top 3 most similar pairs:")
    pairs = []
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            pairs.append((i, j, similarity_matrix[i][j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (idx1, idx2, score) in enumerate(pairs[:3]):
        print(f"  {i+1}. '{sample_texts[idx1]}' ↔ '{sample_texts[idx2]}' (similarity: {score:.4f})")
    
    # Show embeddings dimensionality
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Total samples processed: {len(sample_texts)}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune similarity models for anomaly detection.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    parser.add_argument("--use-hp-search", action="store_true", help="Use hyperparameter search to find optimal parameters.")
    parser.add_argument("--hp-trials", type=int, default=5, help="Number of hyperparameter search trials.")
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
        # Excluded: season (only 1 unique value), care_instructions (only 2 unique values)
    }

    column_configs = {
        'Care Instructions':        {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'colour_name':              {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 3},
        'season':                   {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        'article_structure_name_2': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2},
        # You can add other column configs here if needed
    }

    rules_dir = 'rules'
    
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
            with open(file_path, 'r') as f:
                rules = json.load(f).get('error_rules', [])
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