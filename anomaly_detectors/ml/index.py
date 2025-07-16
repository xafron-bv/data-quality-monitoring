import pandas as pd
import numpy as np
import argparse
import random
import os
import sys
import torch

# Add the parent directory to the path to import the error injection module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from error_injection import load_error_rules

# Import separated modules
from hyperparameter_search import save_aggregated_hp_results, random_hyperparameter_search, get_optimal_parameters
from model_training import train_and_evaluate_similarity_model, get_column_configs, setup_results_directory_structure


def get_rule_to_column_map():
    """
    Get the mapping from rule files to column names.
    """
    return {
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
        
    rule_to_column_map = get_rule_to_column_map()
    column_configs = get_column_configs()

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
        
        # Determine best parameters based on whether HP search is enabled
        if args.use_hp_search:
            best_params, best_recall, best_precision, best_f1, search_results = random_hyperparameter_search(
                df, column_name, rules, device, num_trials=args.hp_trials
            )
            if best_recall <= 0:
                print(f"Hyperparameter search failed for '{column_name}'. Using recall-optimized parameters.")
                best_params = get_optimal_parameters(column_name, config['model'], config['epochs'])
        else:
            # Use recall-optimized parameters
            best_params = get_optimal_parameters(column_name, config['model'], config['epochs'])
            print(f"Using RECALL-OPTIMIZED parameters for '{column_name}'")
            
        train_and_evaluate_similarity_model(
            df, 
            column_name, 
            rules, 
            device=device, 
            best_params=best_params
        )
    
    # Save aggregated hyperparameter search results if HP search was used
    if args.use_hp_search:
        save_aggregated_hp_results()