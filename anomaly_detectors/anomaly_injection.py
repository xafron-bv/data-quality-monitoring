#!/usr/bin/env python3
"""
Anomaly Injection System

This module creates SEMANTIC ANOMALIES for testing anomaly detection systems.
Unlike error_injection.py which creates format/validation errors, this module
creates semantically unusual but technically valid values that should be caught
by pattern-based and ML anomaly detectors.

Examples:
- Color: "Black" -> "Engine Oil Black" (unusual but valid color name)  
- Material: "Cotton" -> "Concrete" (wrong domain but valid text)
- Category: "Blouse" -> "Kitchen Knife Set" (completely wrong category)
"""

import pandas as pd
import numpy as np
import json
import random
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from common.field_mapper import FieldMapper
from common.exceptions import FileOperationError, ConfigurationError


def load_anomaly_rules(rules_path: str) -> List[Dict[str, Any]]:
    """
    Load anomaly injection rules from a JSON file.
    
    Args:
        rules_path: Path to the anomaly injection rules JSON file
        
    Returns:
        List of anomaly injection rules
        
    Raises:
        FileOperationError: If the rules file cannot be read or parsed
    """
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            return rules_data.get('anomaly_rules', [])
    except FileNotFoundError:
        raise FileOperationError(f"Anomaly rules file not found: {rules_path}")
    except json.JSONDecodeError as e:
        raise FileOperationError(f"Invalid JSON in anomaly rules file {rules_path}: {e}")


def apply_anomaly_rule(original_value: Any, rule: Dict[str, Any]) -> Any:
    """
    Apply a single anomaly injection rule to a value.
    
    Args:
        original_value: The original value to apply the anomaly to
        rule: The anomaly injection rule
        
    Returns:
        The value with anomaly applied, or original value if rule doesn't apply
    """
    if pd.isna(original_value):
        return original_value
        
    original_str = str(original_value).strip()
    if not original_str:
        return original_value
    
    # Check conditions if they exist
    conditions = rule.get('conditions', [])
    if conditions:
        for condition in conditions:
            condition_type = condition.get('type')
            condition_value = condition.get('value', '')
            
            if condition_type == 'contains' and condition_value not in original_str:
                return original_value
            # Add more condition types as needed
    
    operation = rule.get('operation')
    params = rule.get('params', {})
    
    if operation == 'value_replacement':
        # Replace entire value with a random anomalous value
        replacement_values = params.get('replacement_values', [])
        if replacement_values:
            return random.choice(replacement_values)
    
    # If no operation matched, return original value
    return original_value


class AnomalyInjector:
    """
    Handles injection of semantic anomalies into datasets for testing anomaly detection systems.
    """
    
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize the anomaly injector with a set of rules.
        
        Args:
            rules: List of anomaly injection rules
        """
        self.rules = rules
        self.anomalies_applied = []
    
    def inject_anomalies(self, df: pd.DataFrame, field_name: str, 
                        max_anomalies: int = 3, anomaly_probability: float = 0.1,
                        field_mapper: Optional[FieldMapper] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Inject anomalies into a DataFrame for a specific field.
        
        Args:
            df: The source DataFrame
            field_name: The field to inject anomalies into
            max_anomalies: Maximum number of anomalies to inject per row
            anomaly_probability: Probability of injecting an anomaly in each row
            field_mapper: Optional field mapper to convert field names to column names
            
        Returns:
            Tuple of (modified_dataframe, list_of_injected_anomalies)
        """
        if not self.rules:
            print(f"No anomaly rules available for field '{field_name}'")
            return df.copy(), []
        
        # Get column name from field name
        if field_mapper:
            try:
                column_name = field_mapper.validate_column_exists(df, field_name)
            except ValueError:
                print(f"Field '{field_name}' not found in DataFrame")
                return df.copy(), []
        else:
            column_name = field_name
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in DataFrame")
                return df.copy(), []
        
        # Create a copy of the DataFrame
        modified_df = df.copy()
        injected_anomalies = []
        
        # Apply anomalies row by row
        for row_idx in range(len(modified_df)):
            if random.random() < anomaly_probability:
                original_value = modified_df.iloc[row_idx][column_name]
                
                # Determine how many anomalies to apply (1 to max_anomalies)
                num_anomalies = random.randint(1, max_anomalies)
                
                current_value = original_value
                anomalies_for_row = []
                
                for _ in range(num_anomalies):
                    # Select a random rule
                    rule = random.choice(self.rules)
                    
                    # Apply the rule
                    new_value = apply_anomaly_rule(current_value, rule)
                    
                    if new_value != current_value:
                        anomalies_for_row.append({
                            'rule_name': rule.get('rule_name', 'unknown'),
                            'rule_description': rule.get('description', ''),
                            'original_value': current_value,
                            'anomalous_value': new_value,
                            'operation': rule.get('operation'),
                            'row_index': row_idx,
                            'column_name': column_name,
                            'field_name': field_name
                        })
                        current_value = new_value
                
                # Update the DataFrame with the final anomalous value
                if current_value != original_value:
                    modified_df.iloc[row_idx, modified_df.columns.get_loc(column_name)] = current_value
                    injected_anomalies.extend(anomalies_for_row)
        
        # Only print if anomalies were actually injected
        # (Let the comprehensive generator handle the summary printing)
        
        return modified_df, injected_anomalies


def generate_anomaly_samples(df: pd.DataFrame, field_name: str, rules: List[Dict[str, Any]], 
                           num_samples: int, max_anomalies: int, anomaly_probability: float,
                           output_dir: str, field_mapper: Optional[FieldMapper] = None) -> List[Dict[str, Any]]:
    """
    Generate multiple samples with injected anomalies for testing anomaly detection.
    
    Args:
        df: Source DataFrame
        field_name: Field to inject anomalies into
        rules: Anomaly injection rules
        num_samples: Number of samples to generate
        max_anomalies: Maximum anomalies per row
        anomaly_probability: Probability of anomaly injection per row
        output_dir: Directory to save samples
        field_mapper: Optional field mapper
        
    Returns:
        List of sample dictionaries with metadata
    """
    print(f"Generating {num_samples} anomaly samples in '{output_dir}' with up to {max_anomalies} anomalies each...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize anomaly injector
    injector = AnomalyInjector(rules)
    
    samples = []
    
    for i in range(num_samples):
        # Generate sample with anomalies
        sample_df, injected_anomalies = injector.inject_anomalies(
            df, field_name, max_anomalies, anomaly_probability, field_mapper
        )
        
        # Set sample name for tracking
        sample_df.name = f"anomaly_sample_{i}"
        
        # Save sample data
        sample_path = os.path.join(output_dir, f"anomaly_sample_{i}.csv")
        sample_df.to_csv(sample_path, index=False)
        
        # Save anomaly metadata
        anomalies_path = os.path.join(output_dir, f"anomaly_sample_{i}_injected_anomalies.json")
        with open(anomalies_path, 'w') as f:
            json.dump(injected_anomalies, f, indent=2, ensure_ascii=False)
        
        # Create sample metadata
        sample_info = {
            "data": sample_df,
            "injected_anomalies": injected_anomalies,
            "sample_index": i,
            "sample_path": sample_path,
            "anomalies_path": anomalies_path,
            "field_name": field_name,
            "num_anomalies": len(injected_anomalies),
            "affected_rows": len(set(a['row_index'] for a in injected_anomalies))
        }
        
        samples.append(sample_info)
    
    # Save summary of all samples
    summary = {
        "field_name": field_name,
        "num_samples": num_samples,
        "total_anomalies": sum(s["num_anomalies"] for s in samples),
        "total_affected_rows": sum(s["affected_rows"] for s in samples),
        "anomaly_probability": anomaly_probability,
        "max_anomalies_per_row": max_anomalies,
        "rules_used": [rule.get('rule_name', 'unknown') for rule in rules],
        "samples": [
            {
                "sample_index": s["sample_index"],
                "sample_path": s["sample_path"],
                "anomalies_path": s["anomalies_path"],
                "num_anomalies": s["num_anomalies"],
                "affected_rows": s["affected_rows"]
            }
            for s in samples
        ]
    }
    
    summary_path = os.path.join(output_dir, "anomaly_samples_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Anomaly sample generation complete.")
    print(f"Total anomalies injected: {summary['total_anomalies']}")
    print(f"Total rows affected: {summary['total_affected_rows']}")
    print(f"Summary saved to: {summary_path}")
    
    return samples


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Anomaly Injection System Test")
    
    # Create test data
    test_df = pd.DataFrame({
        'color_name': ['Black', 'White', 'Red', 'Blue', 'Green'],
        'material': ['Cotton', 'Polyester', 'Silk', 'Wool', 'Nylon']
    })
    
    # Test color anomaly injection
    try:
        color_rules = load_anomaly_rules('anomaly_injection_rules/color_name.json')
        injector = AnomalyInjector(color_rules)
        
        modified_df, anomalies = injector.inject_anomalies(
            test_df, 'color_name', max_anomalies=2, anomaly_probability=0.8
        )
        
        print(f"\nOriginal colors: {list(test_df['color_name'])}")
        print(f"Modified colors: {list(modified_df['color_name'])}")
        print(f"Anomalies injected: {len(anomalies)}")
        
        for anomaly in anomalies:
            print(f"  Row {anomaly['row_index']}: {anomaly['original_value']} -> {anomaly['anomalous_value']}")
        
    except Exception as e:
        print(f"Test failed: {e}") 