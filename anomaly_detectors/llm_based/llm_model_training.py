import argparse
import json
import os
import random
import re
from os import path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'free_gb': total - allocated
        }
    return None


def setup_gpu_memory_management():
    """Setup GPU memory management for RTX 3070 (8GB)."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # Clear any existing cache
    torch.cuda.empty_cache()
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🖥️  GPU: {gpu_name} ({total_memory:.1f}GB)")
    
    # For RTX 3070 (8GB), use conservative memory management
    if total_memory >= 8:
        # Use 70% of available memory to be safe
        memory_fraction = 0.7
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        print(f"🧹 Set memory fraction to {memory_fraction*100:.0f}% ({memory_fraction*total_memory:.1f}GB)")
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends, 'flash_attention') and torch.backends.flash_attention.is_available():
        torch.backends.flash_attention.enable()
        print("⚡ Enabled Flash Attention for memory efficiency")
    
    return torch.device('cuda')


def calculate_safe_batch_size(model_size_mb: int = 500, max_memory_gb: float = 5.6) -> int:
    """
    Calculate safe batch size based on model size and available memory.
    
    Args:
        model_size_mb: Estimated model size in MB
        max_memory_gb: Maximum memory to use in GB (70% of 8GB = 5.6GB)
    
    Returns:
        Safe batch size
    """
    # Conservative estimate: each sample needs ~2x model size for gradients and activations
    memory_per_sample_mb = model_size_mb * 2.5
    
    # Convert to GB
    memory_per_sample_gb = memory_per_sample_mb / 1024
    
    # Calculate safe batch size
    safe_batch_size = int(max_memory_gb / memory_per_sample_gb)
    
    # Ensure minimum and maximum bounds
    safe_batch_size = max(1, min(safe_batch_size, 16))
    
    return safe_batch_size


class TextSequenceDataset(Dataset):
    """Dataset for training language models on text sequences."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # For masked language modeling, we need input_ids and labels
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create labels (same as input_ids for MLM)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def preprocess_text(text: str) -> str:
    """Preprocess text for training."""
    if pd.isna(text):
        return ""

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.\,\%]', '', text)

    return text if text else ""

def analyze_unique_values(df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
    """Analyze unique values in a column to determine suitability for language modeling."""
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found in dataset"}

    # Get unique values
    unique_values = df[column_name].dropna().unique()
    unique_count = len(unique_values)

    # Calculate average length
    text_lengths = [len(str(val)) for val in unique_values]
    avg_length = np.mean(text_lengths) if text_lengths else 0

    # Sample some values for inspection
    sample_values = list(unique_values[:5])

    return {
        "unique_count": unique_count,
        "avg_length": avg_length,
        "sample_values": sample_values,
        "suitable_for_lm": unique_count > 10 and avg_length > 5
    }

def get_model_config(field_name: str) -> Dict[str, Any]:
    """Get model configuration for a specific field."""
    configs = {
        'material': {
            'model_name': 'distilbert-base-uncased',
            'max_length': 128,
            'epochs': 3,
            'batch_size': 4,  # Reduced from 8 for GPU memory safety
            'learning_rate': 2e-5,
            'mask_probability': 0.15
        },
        'care_instructions': {
            'model_name': 'distilbert-base-uncased',
            'max_length': 128,
            'epochs': 3,
            'batch_size': 4,  # Reduced from 8 for GPU memory safety
            'learning_rate': 2e-5,
            'mask_probability': 0.15
        },
        'long_description_nl': {
            'model_name': 'distilbert-base-uncased',
            'max_length': 256,
            'epochs': 2,
            'batch_size': 2,  # Reduced from 4 for GPU memory safety
            'learning_rate': 2e-5,
            'mask_probability': 0.15
        },
        'product_name_en': {
            'model_name': 'distilbert-base-uncased',
            'max_length': 128,
            'epochs': 3,
            'batch_size': 4,  # Reduced from 8 for GPU memory safety
            'learning_rate': 2e-5,
            'mask_probability': 0.15
        }
    }

    # Default configuration
    default_config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'epochs': 2,
        'batch_size': 4,  # Reduced from 8 for GPU memory safety
        'learning_rate': 2e-5,
        'mask_probability': 0.15
    }

    return configs.get(field_name, default_config)

def train_language_model(texts: List[str], field_name: str, config: Dict[str, Any], device: torch.device, output_dir: str) -> Dict[str, Any]:
    """Train a language model on the provided texts."""
    print(f"🤖 Training language model for field '{field_name}'")
    print(f"   📊 Training on {len(texts)} text samples")

    # Load tokenizer and model
    model_name = config['model_name']
    print(f"   📥 Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to device
    model = model.to(device)
    print(f"   📍 Model moved to {device}")

    # Calculate safe batch size based on GPU memory
    if device.type == 'cuda':
        # Estimate model size (DistilBERT is ~260MB)
        model_size_mb = 260 if 'distilbert' in model_name.lower() else 500
        safe_batch_size = calculate_safe_batch_size(model_size_mb)
        config['batch_size'] = min(config['batch_size'], safe_batch_size)
        print(f"   🎯 Using safe batch size: {config['batch_size']} (calculated: {safe_batch_size})")
        
        # Print memory info
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"   💾 GPU Memory: {memory_info['allocated_gb']:.1f}GB allocated, {memory_info['free_gb']:.1f}GB free")

    # Create dataset
    dataset = TextSequenceDataset(texts, tokenizer, config['max_length'])

    # Training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb logging
        gradient_accumulation_steps=max(1, 8 // config['batch_size']),  # Accumulate gradients if batch size is small
        fp16=device.type == 'cuda',  # Use mixed precision on GPU
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        warmup_steps=min(100, len(texts) // config['batch_size']),  # Warmup steps
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train the model with memory monitoring
    print(f"   🚀 Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ⚠️  GPU out of memory error: {e}")
            print(f"   🔄 Trying with smaller batch size...")
            
            # Reduce batch size and try again
            config['batch_size'] = max(1, config['batch_size'] // 2)
            training_args.per_device_train_batch_size = config['batch_size']
            training_args.gradient_accumulation_steps = max(1, 8 // config['batch_size'])
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer
            )
            trainer.train()
        else:
            raise e

    # Clear GPU memory after training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"   🧹 GPU Memory after training: {memory_info['allocated_gb']:.1f}GB allocated")

    # Save the model and tokenizer
    print(f"   💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'output_dir': output_dir
    }

def calculate_sequence_probability(model, tokenizer, text: str, device: torch.device) -> float:
    """Calculate the probability of a text sequence using the trained language model."""
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Calculate average log probability
        probs = torch.softmax(logits, dim=-1)
        input_ids = inputs['input_ids'][0]

        total_log_prob = 0.0
        count = 0

        for i, token_id in enumerate(input_ids):
            if token_id != tokenizer.pad_token_id and token_id != tokenizer.cls_token_id:
                prob = probs[0, i, token_id].item()
                if prob > 0:
                    total_log_prob += torch.log(torch.tensor(prob)).item()
                    count += 1

        # Clear GPU memory
        if device.type == 'cuda':
            del inputs, outputs, logits, probs
            torch.cuda.empty_cache()

        if count > 0:
            return total_log_prob / count
        else:
            return -10.0  # Very low probability

    except Exception as e:
        print(f"Error calculating probability: {e}")
        # Clear GPU memory on error
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return -10.0

def test_anomaly_detection_with_probability(model_info: Dict[str, Any], clean_texts: List[str], field_name: str, threshold: float = -2.0) -> Dict[str, Any]:
    """Test the trained model's ability to detect anomalies using probability scores."""
    print(f"🧪 Testing anomaly detection for field '{field_name}'")

    model = model_info['model']
    tokenizer = model_info['tokenizer']
    device = next(model.parameters()).device

    # Test on normal samples
    normal_samples = random.sample(clean_texts, min(5, len(clean_texts)))
    normal_probabilities = []

    print(f"   📊 Testing on {len(normal_samples)} normal samples:")
    for sample in normal_samples:
        prob = calculate_sequence_probability(model, tokenizer, sample, device)
        normal_probabilities.append(prob)
        print(f"      Normal: '{sample[:50]}...' → {prob:.3f}")

    # Create obvious anomalies for testing
    obvious_anomalies = [
        "INVALID MATERIAL TEXT",
        "Random gibberish text",
        "1234567890",
        "!@#$%^&*()",
        "This is completely wrong material"
    ]

    anomaly_probabilities = []
    print(f"   🚨 Testing on {len(obvious_anomalies)} obvious anomalies:")
    for anomaly in obvious_anomalies:
        prob = calculate_sequence_probability(model, tokenizer, anomaly, device)
        anomaly_probabilities.append(prob)
        print(f"      Anomaly: '{anomaly}' → {prob:.3f}")

    # Calculate metrics
    avg_normal_prob = np.mean(normal_probabilities)
    avg_anomaly_prob = np.mean(anomaly_probabilities)
    probability_separation = avg_normal_prob - avg_anomaly_prob

    # Count detected anomalies
    detected_anomalies = sum(1 for prob in anomaly_probabilities if prob < threshold)
    detection_rate = detected_anomalies / len(obvious_anomalies)

    results = {
        'normal_samples': normal_samples,
        'normal_probabilities': normal_probabilities,
        'anomaly_samples': obvious_anomalies,
        'anomaly_probabilities': anomaly_probabilities,
        'avg_normal_probability': avg_normal_prob,
        'avg_anomaly_probability': avg_anomaly_prob,
        'probability_separation': probability_separation,
        'detection_rate': detection_rate,
        'detected_anomalies': detected_anomalies,
        'total_test_anomalies': len(obvious_anomalies),
        'threshold': threshold
    }

    print(f"   📈 Results:")
    print(f"      Average normal probability: {avg_normal_prob:.3f}")
    print(f"      Average anomaly probability: {avg_anomaly_prob:.3f}")
    print(f"      Probability separation: {probability_separation:.3f}")
    print(f"      Detection rate: {detection_rate:.1%} ({detected_anomalies}/{len(obvious_anomalies)})")

    return results

def setup_output_directory() -> str:
    """Setup output directory for training results."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "llm")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def entry(data_file=None, field=None, epochs=3, batch_size=8, learning_rate=2e-5,
          threshold=-2.0, max_length=128, variation: str = None, device_opt: str = 'auto'):
    """Entry function for LLM model training."""

    if not data_file:
        raise ValueError("data_file is required")
    if not field:
        raise ValueError("field is required")
    if not variation:
        raise ValueError("variation is required for LLM training")

    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Setup output directory
    output_dir = setup_output_directory()

    # Load data
    print(f"📊 Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Get field to column mapping
    field_to_column_map = {
        'category': 'article_structure_name_1',
        'color_name': 'colour_name',
        'material': 'material',
        'care_instructions': 'Care Instructions',
        'season': 'season',
        'size': 'size_name',
        'brand': 'brand',
        'ean': 'EAN',
        'article_number': 'article_number',
        'colour_code': 'colour_code',
        'customs_tariff_number': 'customs_tariff_number',
        'description_short_1': 'description_short_1',
        'long_description_nl': 'long_description_NL',
        'product_name_en': 'product_name_EN'
    }

    column_name = field_to_column_map.get(field)
    if not column_name:
        available_fields = list(field_to_column_map.keys())
        raise ValueError(
            f"Unknown field: {field}\n"
            f"Available fields: {available_fields}"
        )

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in dataset\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Analyze unique values
    print(f"🔍 Analyzing field '{field}' (column: '{column_name}')")
    analysis = analyze_unique_values(df, column_name)

    if "error" in analysis:
        raise RuntimeError(analysis['error'])

    print(f"   📊 Unique values: {analysis['unique_count']}")
    print(f"   📏 Average length: {analysis['avg_length']:.1f} characters")
    print(f"   🎯 Suitable for language modeling: {analysis['suitable_for_lm']}")

    if not analysis['suitable_for_lm']:
        print(f"⚠️  Warning: Field '{field}' may not be suitable for language modeling")
        print(f"   Consider using a field with more unique values and longer text")

    # Get clean texts for training
    clean_texts = []
    for text in df[column_name].dropna():
        processed = preprocess_text(text)
        if processed and len(processed) > 2:  # Skip very short texts
            clean_texts.append(processed)

    if len(clean_texts) < 10:
        raise ValueError(f"Not enough valid texts for training: {len(clean_texts)}. Need at least 10.")

    print(f"   📝 Valid training texts: {len(clean_texts)}")

    # Get model configuration
    config = get_model_config(field)
    config.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length
    })

    # Setup device (auto/cpu/gpu)
    if device_opt == 'cpu':
        device = torch.device('cpu')
    elif device_opt == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # auto
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    print(f"🖥️  Using device: {device}")

    # Train the model under variation-specific directory
    model_output_dir = os.path.join(output_dir, f"{field}_model", variation)
    os.makedirs(model_output_dir, exist_ok=True)
    model_info = train_language_model(clean_texts, field, config, device, model_output_dir)

    # Test anomaly detection
    test_results = test_anomaly_detection_with_probability(
        model_info, clean_texts, field, threshold
    )

    # Save training results
    results_file = os.path.join(output_dir, f"{field}_training_results__{variation}.json")
    results_summary = {
        'field_name': field,
        'variation': variation,
        'column_name': column_name,
        'training_config': config,
        'data_analysis': analysis,
        'training_samples': len(clean_texts),
        'test_results': test_results,
        'model_path': model_output_dir,
        'threshold': threshold
    }

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"💾 Training results saved to: {results_file}")
    print(f"🤖 Model saved to: {model_output_dir}")
    print(f"✅ Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train language model for anomaly detection")
    parser.add_argument("data_file", help="Path to the CSV data file")
    parser.add_argument("--field", required=True, help="Field name to train model for")
    parser.add_argument("--variation", required=True, help="Variation key to train/save the model under")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=-2.0, help="Anomaly detection threshold")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--device", choices=['auto','cpu','gpu'], default='auto', help="Device selection: auto/cpu/gpu")

    args = parser.parse_args()

    entry(
        data_file=args.data_file,
        field=args.field,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        max_length=args.max_length,
        variation=args.variation,
        device_opt=args.device
    )


if __name__ == "__main__":
    main()
