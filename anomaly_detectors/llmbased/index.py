import pandas as pd
import numpy as np
import argparse
import json
import re
import random
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 1. Error Injection Engine ---

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

# --- 2. Data Preparation for Classification ---

def create_augmented_dataset(data_series, rules):
    """
    Creates a large, balanced dataset from the entire data series,
    not just the unique values.
    """
    print("Generating augmented classification dataset from all rows...")
    clean_texts = data_series.dropna().astype(str).tolist()
    
    anomalous_texts = []
    if not rules:
        print("Warning: No rules provided. Cannot generate anomalous data.")
        return [], []

    for text in clean_texts:
        rule = random.choice(rules)
        anomaly = apply_error_rule(text, rule)
        if anomaly == text and len(rules) > 1:
            anomaly = apply_error_rule(text, random.choice(rules))
        anomalous_texts.append(anomaly)

    texts = clean_texts + anomalous_texts
    labels = [0] * len(clean_texts) + [1] * len(anomalous_texts) # 0 for clean, 1 for anomaly
    
    print(f"Dataset created with {len(clean_texts)} clean and {len(anomalous_texts)} anomalous examples.")
    return texts, labels

# --- 3. Model Training and Evaluation ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_and_evaluate_classifier(df, column, rules, device, model_name='distilbert-base-uncased'):
    """Fine-tunes and evaluates a classifier for a specific column on the specified device."""
    
    # 1. Prepare Augmented Dataset
    texts, labels = create_augmented_dataset(df[column], rules)
    if not texts:
        print(f"Could not create a dataset for '{column}'. Skipping.")
        return

    dataset = Dataset.from_dict({'text': texts, 'label': labels}).shuffle(seed=42)
    
    # 2. Tokenize Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 3. Split into Train and Test sets
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.25).values()
    
    # 4. Define Model and move it to the GPU
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Using modern, compatible arguments for TrainingArguments
    # The Trainer will automatically place data on the same device as the model.
    training_args = TrainingArguments(
        output_dir=f'./results_{column.replace(" ", "_").lower()}',
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # 5. Train the Model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # 6. Final Evaluation Results
    print(f"\n--- Best F1-Score achieved for column: '{column}' during training ---")
    
# --- 4. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a classifier for anomaly detection.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
    args = parser.parse_args()
    
    # --- ADDED: Check for GPU and set the device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ GPU found. Using CUDA.")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU found. Using CPU.")

    try:
        df = pd.read_csv(args.csv_file)
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
        except FileNotFoundError: print(f"Warning: Rule file '{file_path}' not found.")
        error_rules_map.setdefault(col, [])

    for column, rules in error_rules_map.items():
        if column not in df.columns or not rules: continue
        print(f"\n{'='*20} Starting Process for Column: {column} {'='*20}")
        train_and_evaluate_classifier(df, column, rules, device=device)