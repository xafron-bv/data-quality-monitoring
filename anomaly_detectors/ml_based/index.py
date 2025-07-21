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

# Import field-to-column mapping
from field_column_map import get_field_to_column_map

# Import anomaly checking functions
from check_anomalies import load_model_for_field, check_anomalies



# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECALL-FOCUSED anomaly detection using sentence transformers.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    parser.add_argument("--use-hp-search", action="store_true", help="Use RECALL-FOCUSED hyperparameter search.")
    parser.add_argument("--hp-trials", type=int, default=15, help="Number of hyperparameter search trials (default: 15).")
    parser.add_argument("--rules", nargs='+', default=None, help="List of field names to include in training/hp search (by field name, space-separated, e.g. 'size material'). If not set, all fields are used.")
    parser.add_argument("--check-anomalies", metavar="FIELD", help="Run anomaly check on the given field using the trained model.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for anomaly detection (default: 0.6)")
    parser.add_argument("--output", default=None, help="Optional output CSV file for anomaly check results.")
    args = parser.parse_args()
    
    if args.check_anomalies:
        field_name = args.check_anomalies
        print(f"Running anomaly check for field '{field_name}'...")
        df = pd.read_csv(args.csv_file)
        model, column_name = load_model_for_field(field_name, results_dir="../results")
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' (mapped from field '{field_name}') not found in CSV.")
        values = df[column_name].tolist()
        results = check_anomalies(model, values, threshold=args.threshold)
        n_anomalies = sum(r['is_anomaly'] for r in results)
        print(f"Checked {len(results)} values in column '{column_name}'. Found {n_anomalies} anomalies.")
        if args.output:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            for r in results[:10]:
                print(r)
        exit(0)
    
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
        
    field_to_column_map = get_field_to_column_map()
    column_configs = get_column_configs()

    rules_dir = '../../error_injection_rules'
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    selected_fields = set(args.rules) if args.rules else None

    for field_name, column_name in field_to_column_map.items():
        # If --rules is set, skip field_names not in the list
        if selected_fields and field_name not in selected_fields:
            continue
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in the CSV. Skipping.")
            continue

        config = column_configs.get(column_name, {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2})

        print(f"\n{'='*20} Starting Process for Column: {column_name} {'='*20}")
        print(f"Using field file: '{field_name}.json', Model: {config['model']}, Epochs: {config['epochs']}")

        if args.use_hp_search:
            print(f"Hyperparameter search enabled with {args.hp_trials} trials")

        file_path = os.path.join(rules_dir, f'{field_name}.json')
        rules = []
        try:
            rules = load_error_rules(file_path)
        except FileNotFoundError:
            print(f"Error: Field file '{file_path}' not found.")
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