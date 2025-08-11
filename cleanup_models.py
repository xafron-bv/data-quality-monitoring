#!/usr/bin/env python3
"""
Script to clean up model directories by removing training artifacts.
This script removes unnecessary files while keeping only the files needed for inference.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_ml_models(models_dir="data/models/ml"):
    """Clean up ML model directories by removing training artifacts."""
    print("🧹 Cleaning up ML models...")
    
    # Required files for ML models (SentenceTransformer)
    required_files = {
        'config.json',
        'model.safetensors', 
        'tokenizer.json',
        'vocab.txt',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'reference_centroid.npy',
        'centroid_metadata.json',
        'modules.json',
        'sentence_bert_config.json',
        'config_sentence_transformers.json'
    }
    
    # Files/directories to remove
    remove_patterns = [
        'checkpoint-*',
        'eval/',
        '1_Pooling/',
        '2_Normalize/',
        '*.csv',  # Evaluation results
        'README.md',  # Training logs
        'training_args.bin'
    ]
    
    trained_dir = os.path.join(models_dir, 'trained')
    if not os.path.exists(trained_dir):
        print(f"❌ Trained models directory not found: {trained_dir}")
        return
    
    total_saved = 0
    total_removed = 0
    
    for field_dir in glob.glob(os.path.join(trained_dir, '*')):
        if not os.path.isdir(field_dir):
            continue
            
        field_name = os.path.basename(field_dir)
        print(f"  📁 Processing field: {field_name}")
        
        for variation_dir in glob.glob(os.path.join(field_dir, '*')):
            if not os.path.isdir(variation_dir):
                continue
                
            variation_name = os.path.basename(variation_dir)
            print(f"    🔄 Processing variation: {variation_name}")
            
            # Remove unnecessary files/directories
            for pattern in remove_patterns:
                matches = glob.glob(os.path.join(variation_dir, pattern))
                for match in matches:
                    if os.path.isdir(match):
                        size = get_dir_size(match)
                        shutil.rmtree(match)
                        total_removed += size
                        print(f"      🗑️  Removed directory: {os.path.basename(match)} ({format_size(size)})")
                    else:
                        size = os.path.getsize(match)
                        os.remove(match)
                        total_removed += size
                        print(f"      🗑️  Removed file: {os.path.basename(match)} ({format_size(size)})")
            
            # Check what's left
            remaining_files = set(os.listdir(variation_dir))
            missing_required = required_files - remaining_files
            
            if missing_required:
                print(f"      ⚠️  Missing required files: {missing_required}")
            else:
                print(f"      ✅ All required files present")
                
            total_saved += get_dir_size(variation_dir)
    
    print(f"\n📊 ML Models Cleanup Summary:")
    print(f"  🗑️  Removed: {format_size(total_removed)}")
    print(f"  💾 Remaining: {format_size(total_saved)}")

def cleanup_llm_models(models_dir="data/models/llm"):
    """Clean up LLM model directories by removing training artifacts."""
    print("\n🧹 Cleaning up LLM models...")
    
    # Required files for LLM models (AutoModelForMaskedLM)
    required_files = {
        'config.json',
        'model.safetensors',
        'tokenizer.json', 
        'vocab.txt',
        'special_tokens_map.json',
        'tokenizer_config.json'
    }
    
    # Files/directories to remove
    remove_patterns = [
        'checkpoint-*',
        'training_args.bin',
        '*.json'  # Training results JSON files (keep only config files)
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ LLM models directory not found: {models_dir}")
        return
    
    total_saved = 0
    total_removed = 0
    
    for model_dir in glob.glob(os.path.join(models_dir, '*_model')):
        if not os.path.isdir(model_dir):
            continue
            
        model_name = os.path.basename(model_dir)
        print(f"  📁 Processing model: {model_name}")
        
        for variation_dir in glob.glob(os.path.join(model_dir, '*')):
            if not os.path.isdir(variation_dir):
                continue
                
            variation_name = os.path.basename(variation_dir)
            print(f"    🔄 Processing variation: {variation_name}")
            
            # Remove unnecessary files/directories
            for pattern in remove_patterns:
                matches = glob.glob(os.path.join(variation_dir, pattern))
                for match in matches:
                    # Don't remove config files
                    if os.path.basename(match) in required_files:
                        continue
                        
                    if os.path.isdir(match):
                        size = get_dir_size(match)
                        shutil.rmtree(match)
                        total_removed += size
                        print(f"      🗑️  Removed directory: {os.path.basename(match)} ({format_size(size)})")
                    else:
                        size = os.path.getsize(match)
                        os.remove(match)
                        total_removed += size
                        print(f"      🗑️  Removed file: {os.path.basename(match)} ({format_size(size)})")
            
            # Check what's left
            remaining_files = set(os.listdir(variation_dir))
            missing_required = required_files - remaining_files
            
            if missing_required:
                print(f"      ⚠️  Missing required files: {missing_required}")
            else:
                print(f"      ✅ All required files present")
                
            total_saved += get_dir_size(variation_dir)
    
    print(f"\n📊 LLM Models Cleanup Summary:")
    print(f"  🗑️  Removed: {format_size(total_removed)}")
    print(f"  💾 Remaining: {format_size(total_saved)}")

def get_dir_size(path):
    """Get the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def main():
    """Main function to clean up all model directories."""
    print("🚀 Starting model cleanup...")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📍 Working directory: {current_dir}")
    
    # Clean up ML models
    ml_models_dir = os.path.join(current_dir, "data", "models", "ml")
    if os.path.exists(ml_models_dir):
        cleanup_ml_models(ml_models_dir)
    else:
        print(f"❌ ML models directory not found: {ml_models_dir}")
    
    # Clean up LLM models
    llm_models_dir = os.path.join(current_dir, "data", "models", "llm")
    if os.path.exists(llm_models_dir):
        cleanup_llm_models(llm_models_dir)
    else:
        print(f"❌ LLM models directory not found: {llm_models_dir}")
    
    # Calculate total savings
    print(f"\n🎯 Cleanup completed!")
    print(f"💡 Models are now optimized for inference only.")
    print(f"💡 You can now zip the models directory for distribution.")

if __name__ == "__main__":
    main()
