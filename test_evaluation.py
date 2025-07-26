#!/usr/bin/env python3
"""
Test script to evaluate ML model performance more comprehensively.
"""

import os
import sys
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from error_injection import load_error_rules, apply_error_rule
from anomaly_detectors.anomaly_injection import load_anomaly_rules, apply_anomaly_rule
from anomaly_detectors.ml_based.model_training import create_improved_triplet_dataset, preprocess_text

def test_model_evaluation(field_name="category"):
    """Test the model evaluation process more comprehensively."""
    print(f"\n{'='*60}")
    print(f"Comprehensive Model Evaluation Test for {field_name}")
    print(f"{'='*60}")
    
    # Load rules
    error_rules_path = f"validators/error_injection_rules/{field_name}.json"
    anomaly_rules_path = f"anomaly_detectors/anomaly_injection_rules/{field_name}.json"
    
    all_rules = []
    
    # Load error rules
    try:
        with open(error_rules_path, 'r') as f:
            error_rules_data = json.load(f)
            error_rules = error_rules_data.get('error_rules', [])
            all_rules.extend(error_rules)
            print(f"✅ Loaded {len(error_rules)} error rules")
    except Exception as e:
        print(f"❌ Failed to load error rules: {e}")
        error_rules = []
    
    # Load anomaly rules
    try:
        anomaly_rules = load_anomaly_rules(anomaly_rules_path)
        # Convert to error rule format
        converted_anomaly_rules = []
        for rule in anomaly_rules:
            converted_rule = {
                'rule_name': rule.get('rule_name', 'unknown'),
                'description': rule.get('description', ''),
                'operation': rule.get('operation', 'value_replacement'),
                'params': rule.get('params', {}),
                'conditions': rule.get('conditions', []),
                'is_anomaly_rule': True
            }
            converted_anomaly_rules.append(converted_rule)
        all_rules.extend(converted_anomaly_rules)
        print(f"✅ Loaded {len(anomaly_rules)} anomaly rules (converted)")
    except Exception as e:
        print(f"❌ Failed to load anomaly rules: {e}")
        anomaly_rules = []
    
    print(f"📊 Total rules: {len(all_rules)}")
    
    # Create test data
    import pandas as pd
    test_data = pd.Series([
        "Blouse", "Dress", "Pants", "Shirt", "Jacket", "Skirt", "Coat", "Sweater",
        "T-shirt", "Jeans", "Suit", "Blazer", "Cardigan", "Hoodie", "Tank Top"
    ])
    
    # Test triplet creation
    print(f"\n🔧 Creating triplets...")
    triplets = create_improved_triplet_dataset(test_data, all_rules, field_name)
    print(f"✅ Created {len(triplets)} triplets")
    
    # Test with a simple model (no training, just to see embeddings)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n📊 Testing with different thresholds:")
    print(f"{'Threshold':<10} {'Clean-Clean':<15} {'Clean-Anomaly':<15} {'Separation':<15}")
    print("-" * 60)
    
    clean_texts = test_data.tolist()
    
    # Calculate clean-to-clean similarities
    clean_clean_similarities = []
    for i in range(len(clean_texts)):
        for j in range(i+1, len(clean_texts)):
            clean_embedding1 = model.encode([clean_texts[i]])
            clean_embedding2 = model.encode([clean_texts[j]])
            similarity = cosine_similarity(clean_embedding1, clean_embedding2)[0][0]
            clean_clean_similarities.append(similarity)
    
    # Calculate clean-to-anomaly similarities
    clean_anomaly_similarities = []
    for clean_text in clean_texts[:5]:  # Test first 5 clean texts
        # Create 3 different anomalies for each clean text
        for _ in range(3):
            rule = random.choice(all_rules)
            anomalous_text = apply_error_rule(clean_text, rule)
            if anomalous_text != clean_text:
                clean_embedding = model.encode([clean_text])
                anomaly_embedding = model.encode([anomalous_text])
                similarity = cosine_similarity(clean_embedding, anomaly_embedding)[0][0]
                clean_anomaly_similarities.append(similarity)
    
    # Calculate statistics
    avg_clean_clean = np.mean(clean_clean_similarities)
    avg_clean_anomaly = np.mean(clean_anomaly_similarities)
    separation = avg_clean_clean - avg_clean_anomaly
    
    print(f"{'AVG':<10} {avg_clean_clean:<15.3f} {avg_clean_anomaly:<15.3f} {separation:<15.3f}")
    
    # Test each threshold
    for threshold in thresholds:
        # Count how many clean pairs would be flagged as anomalous
        clean_flagged = sum(1 for s in clean_clean_similarities if s < threshold)
        clean_total = len(clean_clean_similarities)
        
        # Count how many anomalies would be detected
        anomalies_detected = sum(1 for s in clean_anomaly_similarities if s < threshold)
        anomalies_total = len(clean_anomaly_similarities)
        
        print(f"{threshold:<10.1f} {clean_flagged}/{clean_total:<15} {anomalies_detected}/{anomalies_total:<15} {threshold:<15.1f}")
    
    # Show some examples
    print(f"\n📋 Example similarities:")
    print(f"{'Clean Text':<15} {'Anomalous Text':<25} {'Similarity':<12}")
    print("-" * 55)
    
    for i, clean_text in enumerate(clean_texts[:3]):
        for j in range(2):
            rule = random.choice(all_rules)
            anomalous_text = apply_error_rule(clean_text, rule)
            if anomalous_text != clean_text:
                clean_embedding = model.encode([clean_text])
                anomaly_embedding = model.encode([anomalous_text])
                similarity = cosine_similarity(clean_embedding, anomaly_embedding)[0][0]
                print(f"{clean_text:<15} {anomalous_text[:24]:<25} {similarity:<12.3f}")

def test_original_evaluation_logic():
    """Test the original evaluation logic to see why it gives 100% results."""
    print(f"\n{'='*60}")
    print(f"Testing Original Evaluation Logic")
    print(f"{'='*60}")
    
    # Simulate the original evaluation logic
    field_name = "category"
    
    # Load rules
    error_rules_path = f"validators/error_injection_rules/{field_name}.json"
    anomaly_rules_path = f"anomaly_detectors/anomaly_injection_rules/{field_name}.json"
    
    all_rules = []
    
    # Load error rules
    with open(error_rules_path, 'r') as f:
        error_rules_data = json.load(f)
        error_rules = error_rules_data.get('error_rules', [])
        all_rules.extend(error_rules)
    
    # Load anomaly rules
    anomaly_rules = load_anomaly_rules(anomaly_rules_path)
    converted_anomaly_rules = []
    for rule in anomaly_rules:
        converted_rule = {
            'rule_name': rule.get('rule_name', 'unknown'),
            'description': rule.get('description', ''),
            'operation': rule.get('operation', 'value_replacement'),
            'params': rule.get('params', {}),
            'conditions': rule.get('conditions', []),
            'is_anomaly_rule': True
        }
        converted_anomaly_rules.append(converted_rule)
    all_rules.extend(converted_anomaly_rules)
    
    # Test data
    clean_texts = ["Blouse", "Dress", "Pants", "Shirt", "Jacket"]
    
    # Use recall-optimized threshold
    threshold = 0.6
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test recall: how many actual anomalies are detected
    detected_anomalies = 0
    total_anomalies = 0
    
    print(f"Testing recall with threshold {threshold}:")
    for clean_text in clean_texts:
        # Create corrupted version
        rule = random.choice(all_rules)
        corrupted_text = apply_error_rule(clean_text, rule)
        
        # Preprocess corrupted text
        corrupted_text = preprocess_text(corrupted_text)
        
        if corrupted_text != clean_text:
            # Get embeddings
            clean_embedding = model.encode([clean_text])
            corrupted_embedding = model.encode([corrupted_text])
            
            # Calculate similarity
            similarity = cosine_similarity(clean_embedding, corrupted_embedding)[0][0]
            
            print(f"  '{clean_text}' -> '{corrupted_text}': similarity = {similarity:.3f}")
            
            # Check if anomaly was detected
            if similarity < threshold:
                detected_anomalies += 1
                print(f"    ✅ DETECTED (similarity < {threshold})")
            else:
                print(f"    ❌ NOT DETECTED (similarity >= {threshold})")
            
            total_anomalies += 1
    
    recall = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0
    print(f"Recall: {detected_anomalies}/{total_anomalies} = {recall:.3f}")
    
    # Test precision: how many flagged items are actually anomalies
    clean_pairs_flagged = 0
    total_clean_pairs = 0
    
    print(f"\nTesting precision with threshold {threshold}:")
    # Test random pairs of clean texts
    for i in range(min(3, len(clean_texts))):
        for j in range(i+1, min(i+3, len(clean_texts))):
            clean_text1 = clean_texts[i]
            clean_text2 = clean_texts[j]
            
            # Get embeddings
            clean_embedding1 = model.encode([clean_text1])
            clean_embedding2 = model.encode([clean_text2])
            
            # Calculate similarity
            similarity = cosine_similarity(clean_embedding1, clean_embedding2)[0][0]
            
            print(f"  '{clean_text1}' vs '{clean_text2}': similarity = {similarity:.3f}")
            
            # Check if clean pair was flagged as anomalous (false positive)
            if similarity < threshold:
                clean_pairs_flagged += 1
                print(f"    ❌ FALSE POSITIVE (similarity < {threshold})")
            else:
                print(f"    ✅ CORRECT (similarity >= {threshold})")
            
            total_clean_pairs += 1
    
    precision = detected_anomalies / (detected_anomalies + clean_pairs_flagged) if (detected_anomalies + clean_pairs_flagged) > 0 else 0.0
    print(f"Precision: {detected_anomalies}/({detected_anomalies}+{clean_pairs_flagged}) = {precision:.3f}")

if __name__ == "__main__":
    test_model_evaluation("category")
    test_original_evaluation_logic() 