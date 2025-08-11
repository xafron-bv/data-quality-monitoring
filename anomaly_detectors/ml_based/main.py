"""
ML-based anomaly detection index generation module.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common.anomaly_injection import load_anomaly_rules
from common.brand_config import get_available_brands, load_brand_config

# Add the parent directory to the path to import the error injection module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from common.error_injection import load_error_rules

# Import field-to-column mapping
from common.field_column_map import get_field_to_column_map
from common.brand_config import load_brand_config

# Import anomaly checking functions
from anomaly_detectors.ml_based.check_anomalies import check_anomalies, load_model_for_field

# Import GPU utilities
from anomaly_detectors.ml_based.gpu_utils import get_optimal_device, print_device_info

# Import separated modules
from anomaly_detectors.ml_based.hyperparameter_search import get_optimal_parameters, random_hyperparameter_search, save_aggregated_hp_results
from anomaly_detectors.ml_based.model_training import get_field_configs, setup_results_directory_structure, train_and_evaluate_similarity_model

# --- Main Execution ---

def entry(csv_file=None, use_hp_search=False, hp_trials=15, fields=None, check_anomalies=None,
          threshold=0.6, output=None, brand=None, brand_config=None):
    """Entry function for ML index generation."""

    if not csv_file:
        raise ValueError("csv_file is required")

    # Handle brand configuration
    if not brand:
        available_brands = get_available_brands()
        if len(available_brands) == 1:
            brand = available_brands[0]
            print(f"Using default brand: {brand}")
        else:
            raise ValueError("Brand must be specified with --brand option")

    brand_config_obj = load_brand_config(brand)
    print(f"Using brand configuration: {brand}")

    if check_anomalies:
        field_name = check_anomalies
        print(f"Running anomaly check for field '{field_name}'...")
        df = pd.read_csv(csv_file)
        model, column_name, reference_centroid = load_model_for_field(field_name)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' (mapped from field '{field_name}') not found in CSV.")
        values = df[column_name].tolist()
        anomaly_results = check_anomalies(model, values, threshold=threshold, reference_centroid=reference_centroid)
        n_anomalies = sum(r['is_anomaly'] for r in anomaly_results)
        print(f"Checked {len(anomaly_results)} values in column '{column_name}'. Found {n_anomalies} anomalies.")
        if output:
            out_df = pd.DataFrame(anomaly_results)
            out_df.to_csv(output, index=False)
            print(f"Results saved to {output}")
        else:
            for r in anomaly_results:
              if r['is_anomaly']:
                print(r)
        return

    print("🎯 RECALL-OPTIMIZED Anomaly Detection Training")
    print("💡 Strategy: Better to flag clean data as anomalous than to miss actual anomalies")

    # Setup organized directory structure for all outputs
    setup_results_directory_structure()

    # Determine optimal device using shared utility
    device = get_optimal_device(use_gpu=True)
    print_device_info(device, "ML training")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    field_to_column_map = get_field_to_column_map()
    field_configs = get_field_configs()

    error_rules_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'validators', 'error_injection_rules')
    anomaly_rules_dir = os.path.join(os.path.dirname(__file__), '..', 'anomaly_injection_rules')

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    selected_fields = set(fields) if fields else None

    for field_name, column_name in field_to_column_map.items():
        # If --fields is set, skip field_names not in the list
        if selected_fields and field_name not in selected_fields:
            continue
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in the CSV. Skipping.")
            continue

        config = field_configs.get(field_name, {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'epochs': 2})

        print(f"\n{'='*20} Starting Process for Field: {field_name} (Column: {column_name}) {'='*20}")
        print(f"Using field file: '{field_name}.json', Model: {config['model']}, Epochs: {config['epochs']}")

        if use_hp_search:
            print(f"Hyperparameter search enabled with {hp_trials} trials")

        # Load both error injection rules (format/validation anomalies) and anomaly injection rules (semantic anomalies)
        all_rules = []

        # Load error injection rules (format/validation anomalies)
        try:
            variation = brand_config_obj.field_variations.get(field_name) if hasattr(brand_config_obj, 'field_variations') else None
            if not variation:
                raise ValueError(f"Variation is required for field '{field_name}' to load error injection rules")
            field_dir = os.path.join(error_rules_dir, field_name)
            candidate = os.path.join(field_dir, f"{variation}.json")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Error injection rules not found at {candidate}")
            error_rules = load_error_rules(candidate)
            all_rules.extend(error_rules)
            print(f"Loaded {len(error_rules)} error injection rules from {candidate}")
        except FileNotFoundError:
            pass

        # Load anomaly injection rules (semantic anomalies)
        anomaly_file_path = os.path.join(anomaly_rules_dir, f'{field_name}.json')
        try:
            # Import anomaly injection functions
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from common.anomaly_injection import load_anomaly_rules

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
        field_rules = all_rules

        # Determine best parameters based on whether HP search is enabled
        if use_hp_search:
            best_params, best_recall, best_precision, best_f1, search_results = random_hyperparameter_search(
                df, field_name, column_name, field_rules, device, num_trials=hp_trials
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
            field_rules,
            device=device,
            best_params=best_params,
            variation=brand_config_obj.field_variations[field_name]
        )

    # Save aggregated hyperparameter search results if HP search was used
    if use_hp_search:
        save_aggregated_hp_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECALL-FOCUSED anomaly detection using sentence transformers.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    parser.add_argument("--use-hp-search", action="store_true", help="Use RECALL-FOCUSED hyperparameter search.")
    parser.add_argument("--hp-trials", type=int, default=15, help="Number of hyperparameter search trials (default: 15).")
    parser.add_argument("--fields", nargs='+', default=None, help="List of field names to include in training/hp search (by field name, space-separated, e.g. 'size material'). If not set, all fields are used.")
    parser.add_argument("--check-anomalies", metavar="FIELD", help="Run anomaly check on the given field using the trained model.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for anomaly detection (default: 0.6)")
    parser.add_argument("--output", default=None, help="Optional output CSV file for anomaly check results.")
    parser.add_argument("--brand", help="Brand name (deprecated - uses static config)")
    parser.add_argument("--brand-config", help="Path to brand configuration JSON file (deprecated - uses static config)")
    args = parser.parse_args()

    entry(
        csv_file=args.csv_file,
        use_hp_search=args.use_hp_search,
        hp_trials=args.hp_trials,
        fields=args.fields,
        check_anomalies=args.check_anomalies,
        threshold=args.threshold,
        output=args.output,
        brand=args.brand,
        brand_config=args.brand_config
    )
