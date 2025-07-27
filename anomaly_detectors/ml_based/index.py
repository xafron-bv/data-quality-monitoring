"""
ML-based anomaly detection index generation module.
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
import random
import torch

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from brand_config import load_brand_config, get_available_brands
from anomaly_detectors.anomaly_injection import load_anomaly_rules

# Add the parent directory to the path to import the error injection module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from error_injection import load_error_rules

# Import separated modules
from hyperparameter_search import save_aggregated_hp_results, random_hyperparameter_search, get_optimal_parameters
from model_training import train_and_evaluate_similarity_model, get_field_configs, setup_results_directory_structure

# Import field-to-column mapping
from field_column_map import get_field_to_column_map

# Import anomaly checking functions
from check_anomalies import load_model_for_field, check_anomalies

# Import GPU utilities
from gpu_utils import get_optimal_device, print_device_info



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
    parser.add_argument("--brand", help="Brand name (deprecated - uses static config)")
    parser.add_argument("--brand-config", help="Path to brand configuration JSON file (deprecated - uses static config)")
    args = parser.parse_args()
    
    # Handle brand configuration
    if not args.brand:
        available_brands = get_available_brands()
        if len(available_brands) == 1:
            args.brand = available_brands[0]
            print(f"Using default brand: {args.brand}")
        else:
            print("Error: Brand must be specified with --brand")
            sys.exit(1)
    
    brand_config = load_brand_config(args.brand)
    print(f"Using brand configuration: {args.brand}")
    
    if args.check_anomalies:
        field_name = args.check_anomalies
        print(f"Running anomaly check for field '{field_name}'...")
        df = pd.read_csv(args.csv_file)
        model, column_name, reference_centroid = load_model_for_field(field_name, results_dir=os.path.join('..', 'results'))
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' (mapped from field '{field_name}') not found in CSV.")
        values = df[column_name].tolist()
        results = check_anomalies(model, values, threshold=args.threshold, reference_centroid=reference_centroid)
        n_anomalies = sum(r['is_anomaly'] for r in results)
        print(f"Checked {len(results)} values in column '{column_name}'. Found {n_anomalies} anomalies.")
        if args.output:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            for r in results:
              if r['is_anomaly']:
                print(r)
        exit(0)
    
    print("🎯 RECALL-OPTIMIZED Anomaly Detection Training")
    print("💡 Strategy: Better to flag clean data as anomalous than to miss actual anomalies")
    
    # Setup organized directory structure for all outputs
    setup_results_directory_structure()
    
    # Determine optimal device using shared utility
    device = get_optimal_device(use_gpu=True)
    print_device_info(device, "ML training")

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}"); exit()
        
    field_to_column_map = get_field_to_column_map()
    field_configs = get_field_configs()

    error_rules_dir = os.path.join('..', '..', 'validators', 'error_injection_rules')
    anomaly_rules_dir = os.path.join(os.path.dirname(__file__), '..', 'anomaly_injection_rules')
    
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

        config = field_configs.get(field_name, {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2})

        print(f"\n{'='*20} Starting Process for Field: {field_name} (Column: {column_name}) {'='*20}")
        print(f"Using field file: '{field_name}.json', Model: {config['model']}, Epochs: {config['epochs']}")

        if args.use_hp_search:
            print(f"Hyperparameter search enabled with {args.hp_trials} trials")

        # Load both error injection rules (format/validation anomalies) and anomaly injection rules (semantic anomalies)
        all_rules = []
        
        # Load error injection rules (format/validation anomalies)
        error_file_path = os.path.join(error_rules_dir, f'{field_name}.json')
        try:
            error_rules = load_error_rules(error_file_path)
            all_rules.extend(error_rules)
            print(f"Loaded {len(error_rules)} error injection rules from {error_file_path}")
        except FileNotFoundError:
            print(f"Warning: Error injection rules file '{error_file_path}' not found.")
        
        # Load anomaly injection rules (semantic anomalies)
        anomaly_file_path = os.path.join(anomaly_rules_dir, f'{field_name}.json')
        try:
            # Import anomaly injection functions
            import sys
            sys.path.append(os.path.join('..', '..'))
            from anomaly_detectors.anomaly_injection import load_anomaly_rules
            
            anomaly_rules = load_anomaly_rules(anomaly_file_path)
            # Convert anomaly rules to error rule format for compatibility
            converted_anomaly_rules = []
            for rule in anomaly_rules:
                # Convert anomaly rule to error rule format
                converted_rule = {
                    'rule_name': rule.get('rule_name', 'unknown'),
                    'description': rule.get('description', ''),
                    'operation': rule.get('operation', 'value_replacement'),
                    'params': rule.get('params', {}),
                    'conditions': rule.get('conditions', []),
                    'is_anomaly_rule': True  # Flag to identify converted rules
                }
                converted_anomaly_rules.append(converted_rule)
            
            all_rules.extend(converted_anomaly_rules)
            print(f"Loaded {len(anomaly_rules)} anomaly injection rules from {anomaly_file_path}")
        except FileNotFoundError:
            print(f"Warning: Anomaly injection rules file '{anomaly_file_path}' not found.")
        except Exception as e:
            print(f"Warning: Failed to load anomaly injection rules: {e}")

        if not all_rules:
            print(f"No rules found for field '{field_name}'. Skipping.")
            continue
        
        print(f"Total rules for training: {len(all_rules)} (errors + anomalies)")
        rules = all_rules

        # Determine best parameters based on whether HP search is enabled
        if args.use_hp_search:
            best_params, best_recall, best_precision, best_f1, search_results = random_hyperparameter_search(
                df, field_name, column_name, rules, device, num_trials=args.hp_trials
            )
            if best_recall <= 0:
                print(f"Hyperparameter search failed for field '{field_name}'. Using recall-optimized parameters.")
                best_params = get_optimal_parameters(field_name, config['model'], config['epochs'])
        else:
            # Use recall-optimized parameters
            best_params = get_optimal_parameters(field_name, config['model'], config['epochs'])
            print(f"Using RECALL-OPTIMIZED parameters for field '{field_name}'")

        train_and_evaluate_similarity_model(
            df,
            field_name,
            column_name,
            rules,
            device=device,
            best_params=best_params
        )
    
    # Save aggregated hyperparameter search results if HP search was used
    if args.use_hp_search:
        save_aggregated_hp_results()