import pandas as pd
import numpy as np
import argparse
import json
import re
import random
import os
import torch
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sklearn.ensemble import IsolationForest
from collections import defaultdict

# --- 1. Error Injection Engine ---

def apply_error_rule(value, rule):
    if 'conditions' in rule and rule['conditions']:
        should_apply = False
        for cond in rule['conditions']:
            if cond['type'] == 'contains' and cond['value'] in str(value):
                should_apply = True; break
        if not should_apply: return value
    op, params = rule['operation'], rule.get('params', {})
    val_str = str(value)
    if op == 'string_replace': return val_str.replace(params['find'], params['replace'])
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

def create_anomalous_dataset(df, error_rules_map, anomaly_fraction=0.15):
    print(f"\nInjecting anomalies into {anomaly_fraction:.0%} of the data for testing...")
    anomalous_df = df.copy()
    anomalous_df['is_anomaly'] = False
    anomaly_indices = df.sample(frac=anomaly_fraction, random_state=42).index
    anomalous_df.loc[anomaly_indices, 'is_anomaly'] = True
    for col_name, rules in error_rules_map.items():
        if col_name not in anomalous_df.columns or not rules: continue
        col_indices = anomalous_df[anomalous_df['is_anomaly']].index
        for idx in col_indices:
            if not rules: continue
            rule = random.choice(rules)
            original_value = anomalous_df.loc[idx, col_name]
            modified_value = apply_error_rule(original_value, rule)
            if modified_value != original_value:
                anomalous_df.loc[idx, col_name] = modified_value
    print(f"Injection complete. {len(anomaly_indices)} rows were marked for corruption.")
    return anomalous_df

# --- 2. Fine-Tuning Stage ---

def create_triplets(clean_values, rules, num_triplets=5000):
    triplets = []
    unique_values = list(clean_values)
    if len(unique_values) < 2: return []
    print(f"Generating {num_triplets} triplets...")
    for _ in range(num_triplets):
        anchor, positive = random.sample(unique_values, 2)
        if not rules: continue
        rule = random.choice(rules)
        negative = apply_error_rule(anchor, rule)
        if negative != anchor:
            triplets.append(InputExample(texts=[anchor, positive, negative]))
    return triplets

## GPU-FAST: Add 'device' argument and increase default batch size for GPU performance.
def fine_tune_model(df, error_rules_map, device, base_model='bert-base-uncased', epochs=2, batch_size=128):
    tuned_model_paths = {}
    for column, rules in error_rules_map.items():
        if column not in df.columns or not rules: continue
        print(f"\n--- Fine-tuning model for column: '{column}' ---")
        output_path = f'tuned_model_{column.replace(" ", "_").lower()}'
        clean_values = df[column].dropna().unique()
        train_examples = create_triplets(clean_values, rules)
        if not train_examples:
            print(f"Not enough unique values/rules in '{column}'. Skipping fine-tuning.")
            continue
        word_embedding_model = models.Transformer(base_model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        ## GPU-FAST: Initialize the model directly on the specified device ('cuda' or 'cpu').
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        train_loss = losses.TripletLoss(model=model)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, show_progress_bar=True)
        model.save(output_path)
        tuned_model_paths[column] = output_path
        print(f"Fine-tuned model saved to: {output_path}")
    return tuned_model_paths

# --- 3. Main Execution and Evaluation ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a semantic anomaly detector.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    args = parser.parse_args()

    ## GPU-FAST: Automatically detect and set the device for PyTorch.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device: {device}")

    try:
        clean_df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}"); exit()

    rule_files = {
        'Care Instructions': 'care_instructions.json', 'colour_name': 'color_name.json',
        'material': 'material.json', 'season': 'season.json', 'size_name': 'size.json'
    }
    rules_dir = 'rules'
    error_rules_map = {}
    for col, file_name in rule_files.items():
        file_path = os.path.join(rules_dir, file_name)
        try:
            with open(file_path, 'r') as f: error_rules_map[col] = json.load(f)['error_rules']
        except FileNotFoundError: print(f"Warning: Rule file '{file_path}' not found. Assigning empty rules.")
        error_rules_map.setdefault(col, [])


    # STAGE 1: FINE-TUNE THE MODELS
    ## GPU-FAST: Pass the detected device to the fine-tuning function.
    tuned_model_paths = fine_tune_model(clean_df, error_rules_map, device=device, epochs=2)

    # STAGE 2: EVALUATE THE FINE-TUNED MODELS
    train_df = clean_df.sample(frac=0.7, random_state=42)
    test_df_clean = clean_df.drop(train_df.index)

    anomalous_test_df = create_anomalous_dataset(test_df_clean, error_rules_map, anomaly_fraction=0.5)
    true_anomalies = set(anomalous_test_df[anomalous_test_df['is_anomaly']].index)

    for column, model_path in tuned_model_paths.items():
        if not os.path.exists(model_path):
            print(f"\n--- Skipping evaluation for column: '{column}' (no model was trained) ---")
            continue

        print(f"\n--- Evaluating fine-tuned model for column: '{column}' ---")

        ## GPU-FAST: Load the model onto the specified device.
        tuned_model = SentenceTransformer(model_path, device=device)

        # Train the Isolation Forest detector ONLY on clean data embeddings
        clean_train_values = train_df[column].dropna().unique()
        ## GPU-FAST: Encode on the GPU with a larger batch size and visible progress bar.
        clean_embeddings = tuned_model.encode(clean_train_values, show_progress_bar=True, batch_size=256)

        # This part remains on the CPU as scikit-learn does not use the GPU.
        detector = IsolationForest(contamination=0.01, random_state=42).fit(clean_embeddings)

        # Generate embeddings for the entire test set (clean + anomalous)
        test_values = anomalous_test_df[column].dropna().astype(str).tolist()
        ## GPU-FAST: Encode on the GPU with a larger batch size and visible progress bar.
        test_embeddings = tuned_model.encode(test_values, show_progress_bar=True, batch_size=256)

        # Predict which ones are anomalies (-1 means anomaly, 1 means normal)
        predictions = detector.predict(test_embeddings)

        # Get the indices of the predicted anomalies
        results_df = anomalous_test_df[anomalous_test_df[column].isin(test_values)].copy()
        results_df['prediction'] = predictions
        predicted_anomalies = set(results_df[results_df['prediction'] == -1].index)

        # Calculate and print metrics
        tp = len(true_anomalies.intersection(predicted_anomalies))
        fp = len(predicted_anomalies.difference(true_anomalies))
        fn = len(true_anomalies.difference(predicted_anomalies))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("Detector Performance:")
        print(f"  - True Positives (found injected anomalies):    {tp}")
        print(f"  - False Positives (flagged clean data):         {fp}")
        print(f"  - False Negatives (missed injected anomalies):  {fn}")
        print("---")
        print(f"  - Precision: {precision:.2f}")
        print(f"  - Recall:    {recall:.2f}")
        print(f"  - F1-Score:  {f1_score:.2f}")