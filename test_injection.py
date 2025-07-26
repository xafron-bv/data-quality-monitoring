#!/usr/bin/env python3
"""
Test script to verify error and anomaly injection is working correctly.
"""

import os
import sys
import json
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from error_injection import load_error_rules, apply_error_rule
from anomaly_detectors.anomaly_injection import load_anomaly_rules, apply_anomaly_rule

def test_injection_for_field(field_name):
    """Test error and anomaly injection for a specific field."""
    print(f"\n{'='*50}")
    print(f"Testing injection for field: {field_name}")
    print(f"{'='*50}")
    
    # Test data
    test_values = ["Blouse", "Dress", "Pants", "Shirt", "Jacket"]
    
    # Load error injection rules
    error_rules_path = f"validators/error_injection_rules/{field_name}.json"
    try:
        with open(error_rules_path, 'r') as f:
            error_rules_data = json.load(f)
            error_rules = error_rules_data.get('error_rules', [])
        print(f"✅ Loaded {len(error_rules)} error injection rules")
    except Exception as e:
        print(f"❌ Failed to load error rules: {e}")
        error_rules = []
    
    # Load anomaly injection rules
    anomaly_rules_path = f"anomaly_detectors/anomaly_injection_rules/{field_name}.json"
    try:
        anomaly_rules = load_anomaly_rules(anomaly_rules_path)
        print(f"✅ Loaded {len(anomaly_rules)} anomaly injection rules")
    except Exception as e:
        print(f"❌ Failed to load anomaly rules: {e}")
        anomaly_rules = []
    
    print(f"\n📊 Testing with {len(test_values)} sample values:")
    for i, value in enumerate(test_values, 1):
        print(f"\n{i}. Original: '{value}'")
        
        # Test error injection
        if error_rules:
            print("   Error injection results:")
            for j, rule in enumerate(error_rules[:3], 1):  # Test first 3 rules
                try:
                    corrupted = apply_error_rule(value, rule)
                    if corrupted != value:
                        print(f"     Rule {j} ({rule.get('rule_name', 'unnamed')}): '{corrupted}'")
                    else:
                        print(f"     Rule {j} ({rule.get('rule_name', 'unnamed')}): No change")
                except Exception as e:
                    print(f"     Rule {j} failed: {e}")
        
        # Test anomaly injection
        if anomaly_rules:
            print("   Anomaly injection results:")
            for j, rule in enumerate(anomaly_rules[:3], 1):  # Test first 3 rules
                try:
                    anomalous = apply_anomaly_rule(value, rule)
                    if anomalous != value:
                        print(f"     Rule {j} ({rule.get('rule_name', 'unnamed')}): '{anomalous}'")
                    else:
                        print(f"     Rule {j} ({rule.get('rule_name', 'unnamed')}): No change")
                except Exception as e:
                    print(f"     Rule {j} failed: {e}")

def test_triplet_creation():
    """Test the triplet creation process to see if it's working correctly."""
    print(f"\n{'='*50}")
    print("Testing triplet creation process")
    print(f"{'='*50}")
    
    # Import the triplet creation function
    from anomaly_detectors.ml_based.model_training import create_improved_triplet_dataset
    
    # Load rules for category
    error_rules_path = "validators/error_injection_rules/category.json"
    anomaly_rules_path = "anomaly_detectors/anomaly_injection_rules/category.json"
    
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
    test_data = pd.Series(["Blouse", "Dress", "Pants", "Shirt", "Jacket", "Skirt", "Coat", "Sweater"])
    
    # Test triplet creation
    try:
        triplets = create_improved_triplet_dataset(test_data, all_rules, "category")
        print(f"✅ Created {len(triplets)} triplets")
        
        # Show some examples
        print("\n📋 Sample triplets:")
        for i, triplet in enumerate(triplets[:3], 1):
            print(f"  {i}. Anchor: '{triplet.texts[0]}'")
            print(f"     Positive: '{triplet.texts[1]}'")
            print(f"     Negative: '{triplet.texts[2]}'")
            print()
            
    except Exception as e:
        print(f"❌ Triplet creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test injection for category field
    test_injection_for_field("category")
    
    # Test triplet creation
    test_triplet_creation() 