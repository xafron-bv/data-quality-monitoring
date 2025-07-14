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

# --- Error Injection and Data Prep Functions ---

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

def create_augmented_dataset(data_series, rules):
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
    labels = [0] * len(clean_texts) + [1] * len(anomalous_texts)
    print(f"Dataset created with {len(clean_texts)} clean and {len(anomalous_texts)} anomalous examples.")
    return texts, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Model Training Function ---

def train_and_evaluate_classifier(df, column, rules, device, model_name, num_epochs):
    """Fine-tunes and evaluates a classifier with configurable model and epochs."""
    
    texts, labels = create_augmented_dataset(df[column], rules)
    if not texts:
        print(f"Could not create a dataset for '{column}'. Skipping."); return

    dataset = Dataset.from_dict({'text': texts, 'label': labels}).shuffle(seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.25).values()
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # MODIFIED: Arguments compatible with older transformers versions
    training_args = TrainingArguments(
        output_dir=f'./results_{column.replace(" ", "_").lower()}',
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        # The modern arguments below are removed to prevent the error
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    print(f"\n--- Final Performance for column: '{column}' ---")
    final_metrics = trainer.evaluate()
    print(f"  - Precision: {final_metrics.get('eval_precision', 0):.2f}")
    print(f"  - Recall:    {final_metrics.get('eval_recall', 0):.2f}")
    print(f"  - F1-Score:  {final_metrics.get('eval_f1', 0):.2f}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a classifier for anomaly detection.")
    parser.add_argument("csv_file", help="The path to the input CSV file.")
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
        
    column_configs = {
        'Care Instructions': {'model': 'distilbert-base-uncased', 'epochs': 2},
        'colour_name':       {'model': 'distilbert-base-uncased', 'epochs': 3},
        'material':          {'model': 'distilbert-base-uncased', 'epochs': 2},
        'season':            {'model': 'distilbert-base-uncased', 'epochs': 2},
        'size_name':         {'model': 'bert-base-uncased', 'epochs': 3},
    }

    rules_dir = 'rules'
    for column, config in column_configs.items():
        if column not in df.columns: continue
        
        print(f"\n{'='*20} Starting Process for Column: {column} {'='*20}")
        print(f"Using model: {config['model']}, Training for: {config['epochs']} epochs")
        
        file_name = f'{column.lower().replace(" ", "_")}.json'
        file_path = os.path.join(rules_dir, file_name)
        rules = []
        try:
            with open(file_path, 'r') as f:
                rules = json.load(f).get('error_rules', [])
        except FileNotFoundError:
            print(f"Warning: Rule file '{file_path}' not found.")
        
        if not rules:
            print(f"No rules found for '{column}'. Skipping.")
            continue
            
        train_and_evaluate_classifier(
            df, 
            column, 
            rules, 
            device=device, 
            model_name=config['model'],
            num_epochs=config['epochs']
        )