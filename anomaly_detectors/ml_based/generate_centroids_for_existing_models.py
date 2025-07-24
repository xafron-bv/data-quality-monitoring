#!/usr/bin/env python3
"""
Generate Reference Centroids for Existing Trained Models

This script generates reference centroids for all existing trained models
to enable production-ready single-value anomaly detection.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from field_column_map import get_field_to_column_map
from anomaly_detectors.ml_based.model_training import preprocess_text
from anomaly_detectors.ml_based.gpu_utils import get_optimal_device, print_device_info


def generate_centroid_for_model(model_dir, field_name, column_name, df):
    """
    Generate and save reference centroid for a single model.
    
    Args:
        model_dir: Path to the model directory
        field_name: Field name for the model
        column_name: Column name in the dataset
        df: DataFrame containing the training data
    """
    print(f"\n🔄 Processing {field_name} (column: {column_name})")
    
    # Check if centroid already exists
    centroid_path = os.path.join(model_dir, "reference_centroid.npy")
    metadata_path = os.path.join(model_dir, "centroid_metadata.json")
    
    if os.path.exists(centroid_path):
        print(f"   ✅ Reference centroid already exists, skipping...")
        return True
    
    try:
        # Load the model
        device = get_optimal_device(use_gpu=True)
        model = SentenceTransformer(model_dir, device=device)
        print(f"   📚 Loaded model from {os.path.basename(model_dir)}")
        
        # Get clean training texts
        if column_name not in df.columns:
            print(f"   ❌ Column '{column_name}' not found in dataset")
            return False
        
        # Clean and preprocess the data
        clean_texts = df[column_name].dropna().apply(preprocess_text).astype(str).tolist()
        clean_texts = [text for text in clean_texts if text and len(text.strip()) > 0]
        
        if len(clean_texts) < 10:
            print(f"   ⚠️  Warning: Only {len(clean_texts)} clean texts available")
        
        print(f"   📊 Computing centroid from {len(clean_texts)} clean samples...")
        
        # Compute embeddings for clean training data
        clean_embeddings = model.encode(clean_texts, 
                                       batch_size=32, 
                                       show_progress_bar=True,
                                       convert_to_numpy=True)
        
        # Compute reference centroid
        reference_centroid = np.mean(clean_embeddings, axis=0)
        
        # Save centroid as numpy array
        np.save(centroid_path, reference_centroid)
        
        # Save centroid metadata
        centroid_metadata = {
            "num_samples": len(clean_texts),
            "embedding_dim": reference_centroid.shape[0],
            "created_at": datetime.now().isoformat(),
            "field_name": field_name,
            "column_name": column_name,
            "script_version": "generate_centroids_for_existing_models.py"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(centroid_metadata, f, indent=2, default=str)
        
        print(f"   ✅ Saved reference centroid:")
        print(f"      📄 Centroid: {os.path.basename(centroid_path)} (shape: {reference_centroid.shape})")
        print(f"      📄 Metadata: {os.path.basename(metadata_path)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing {field_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to generate centroids for all existing models."""
    print("🚀 Generating Reference Centroids for Existing Trained Models")
    print("=" * 70)
    
    # Load the dataset
    data_file = "../../data/esqualo_2022_fall.csv"
    if not os.path.exists(data_file):
        print(f"❌ Dataset not found: {data_file}")
        print("Please ensure the dataset file exists and update the path if needed.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(data_file)
        print(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Get field to column mapping
    field_to_column = get_field_to_column_map()
    
    # Find all existing model directories
    results_dir = "../results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    model_dirs = []
    for item in os.listdir(results_dir):
        if item.startswith("results_") and os.path.isdir(os.path.join(results_dir, item)):
            model_dirs.append(item)
    
    if not model_dirs:
        print(f"❌ No trained models found in {results_dir}")
        sys.exit(1)
    
    print(f"📋 Found {len(model_dirs)} trained models")
    
    # Process each model
    success_count = 0
    for model_dir_name in sorted(model_dirs):
        # Extract field name from directory name
        field_name = model_dir_name.replace("results_", "").replace("_", " ")
        
        # Find matching field in mapping
        matched_field = None
        for field, column in field_to_column.items():
            if field.replace(" ", "_").lower() == field_name.replace(" ", "_").lower():
                matched_field = field
                break
        
        if not matched_field:
            print(f"\n⚠️  Skipping {model_dir_name}: No matching field found in mapping")
            continue
        
        column_name = field_to_column[matched_field]
        model_dir_path = os.path.join(results_dir, model_dir_name)
        
        # Check if this is a valid model directory
        if not os.path.exists(os.path.join(model_dir_path, "config.json")):
            print(f"\n⚠️  Skipping {model_dir_name}: Not a valid model directory")
            continue
        
        # Generate centroid
        if generate_centroid_for_model(model_dir_path, matched_field, column_name, df):
            success_count += 1
    
    # Summary
    print(f"\n🎉 Centroid Generation Complete!")
    print(f"✅ Successfully processed: {success_count}/{len(model_dirs)} models")
    
    if success_count < len(model_dirs):
        print(f"⚠️  Some models failed to process. Check the output above for details.")
    
    print(f"\n🔧 Models are now ready for production single-value anomaly detection!")


if __name__ == "__main__":
    main() 