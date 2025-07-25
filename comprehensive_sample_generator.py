#!/usr/bin/env python3
"""
Comprehensive Sample Generator

This module creates a single comprehensive sample with errors and anomalies
injected across ALL available fields, rather than generating multiple samples
for a single field. This approach is more realistic for testing data quality
monitoring systems as it simulates real-world scenarios where multiple
fields may have issues simultaneously.
"""

import pandas as pd
import json
import os
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from common_interfaces import FieldMapper
from error_injection import ErrorInjector, load_error_rules
from anomaly_detectors.anomaly_injection import AnomalyInjector, load_anomaly_rules
from exceptions import FileOperationError, ConfigurationError


def get_available_injection_fields(error_rules_dir: str = os.path.join(os.path.dirname(__file__), 'validators', 'error_injection_rules'), 
                                 anomaly_rules_dir: str = os.path.join(os.path.dirname(__file__), 'anomaly_detectors', 'anomaly_injection_rules')) -> Dict[str, Dict[str, bool]]:
    """
    Get fields that have error or anomaly injection rules available.
    
    Returns:
        Dict mapping field names to their available injection types:
        {
            "material": {"errors": True, "anomalies": True},
            "color_name": {"errors": True, "anomalies": True},
            "category": {"errors": True, "anomalies": False},
            ...
        }
    """
    field_mapper = FieldMapper.from_default_mapping()
    available_fields = {}
    
    for field_name in field_mapper.get_available_fields():
        error_rules_path = os.path.join(error_rules_dir, f"{field_name}.json")
        anomaly_rules_path = os.path.join(anomaly_rules_dir, f"{field_name}.json")
        
        available_fields[field_name] = {
            "errors": os.path.exists(error_rules_path),
            "anomalies": os.path.exists(anomaly_rules_path)
        }
    
    return available_fields


def generate_comprehensive_sample(df: pd.DataFrame, 
                                injection_intensity: float = 0.2,
                                max_issues_per_row: int = 2,
                                field_mapper: Optional[FieldMapper] = None,
                                error_rules_dir: str = os.path.join(os.path.dirname(__file__), 'validators', 'error_injection_rules'),
                                anomaly_rules_dir: str = os.path.join(os.path.dirname(__file__), 'anomaly_detectors', 'anomaly_injection_rules')) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Generate a comprehensive sample with errors and anomalies across all available fields.
    
    Args:
        df: Source DataFrame
        injection_intensity: Probability of injecting issues in each cell (0.0-1.0)
        max_issues_per_row: Maximum number of fields to corrupt per row
        field_mapper: Optional field mapper
        error_rules_dir: Directory containing error injection rules
        anomaly_rules_dir: Directory containing anomaly injection rules
        
    Returns:
        Tuple of (corrupted_dataframe, injection_metadata)
        injection_metadata format:
        {
            "field_name": [
                {
                    "row_index": 42,
                    "original_value": "Cotton",
                    "corrupted_value": "Concrete",
                    "injection_type": "anomaly",  # "error" or "anomaly"
                    "rule_name": "construction_materials",
                    "message": "Anomaly: Construction material in clothing field"
                }
            ]
        }
    """
    if field_mapper is None:
        field_mapper = FieldMapper.from_default_mapping()
    
    # Get available fields for injection
    available_fields = get_available_injection_fields(error_rules_dir, anomaly_rules_dir)
    injectable_fields = {
        field: info for field, info in available_fields.items() 
        if info["errors"] or info["anomalies"]
    }
    
    if not injectable_fields:
        print("❌ No fields with injection rules found")
        return df.copy(), {}
    
    print(f"🎯 Generating comprehensive sample with {injection_intensity*100:.1f}% injection intensity")
    print(f"📋 Available fields for injection: {len(injectable_fields)}")
    
    # Validate that fields exist in DataFrame
    valid_fields = {}
    for field_name, info in injectable_fields.items():
        try:
            column_name = field_mapper.validate_column_exists(df, field_name)
            valid_fields[field_name] = {**info, "column_name": column_name}
        except ValueError:
            print(f"   ⚠️  Skipping {field_name}: column not found in DataFrame")
    
    if not valid_fields:
        print("❌ No valid fields found in DataFrame")
        return df.copy(), {}
    
    print(f"✅ Will inject into {len(valid_fields)} valid fields: {list(valid_fields.keys())}")
    
    # Create a copy of the DataFrame for modification
    corrupted_df = df.copy()
    injection_metadata = {}
    
    # Load all injection rules upfront
    field_injectors = {}
    for field_name, info in valid_fields.items():
        injectors = {}
        
        # Load error injection rules if available
        if info["errors"]:
            try:
                error_rules_path = os.path.join(error_rules_dir, f"{field_name}.json")
                error_rules = load_error_rules(error_rules_path)
                injectors["error"] = ErrorInjector(error_rules, field_mapper)
                print(f"   📝 Loaded {len(error_rules)} error rules for {field_name}")
            except Exception as e:
                print(f"   ⚠️  Could not load error rules for {field_name}: {e}")
        
        # Load anomaly injection rules if available  
        if info["anomalies"]:
            try:
                anomaly_rules_path = os.path.join(anomaly_rules_dir, f"{field_name}.json")
                anomaly_rules = load_anomaly_rules(anomaly_rules_path)
                injectors["anomaly"] = AnomalyInjector(anomaly_rules)
                print(f"   🔍 Loaded {len(anomaly_rules)} anomaly rules for {field_name}")
            except Exception as e:
                print(f"   ⚠️  Could not load anomaly rules for {field_name}: {e}")
        
        if injectors:
            field_injectors[field_name] = injectors
    
    # Inject issues row by row
    total_injections = 0
    affected_rows = 0
    
    for row_idx in range(len(corrupted_df)):
        # Determine if this row should have issues injected
        if random.random() > injection_intensity:
            continue
        
        # Randomly select fields to corrupt in this row (max_issues_per_row limit)
        available_fields_for_row = list(field_injectors.keys())
        num_fields_to_corrupt = min(
            random.randint(1, max_issues_per_row),
            len(available_fields_for_row)
        )
        fields_to_corrupt = random.sample(available_fields_for_row, num_fields_to_corrupt)
        
        row_had_injection = False
        
        for field_name in fields_to_corrupt:
            column_name = valid_fields[field_name]["column_name"]
            original_value = corrupted_df.at[row_idx, column_name]
            
            # Skip if original value is null/empty
            if pd.isna(original_value) or str(original_value).strip() == "":
                continue
            
            # Choose injection type (prefer anomalies over errors since errors often fail)
            available_injectors = field_injectors[field_name]
            if "error" in available_injectors and "anomaly" in available_injectors:
                # 30% chance for errors, 70% for anomalies (reverse the previous ratio)
                # This is because error rules are very specific and often don't match
                injection_type = "error" if random.random() < 0.3 else "anomaly"
            elif "error" in available_injectors:
                injection_type = "error"
            else:
                injection_type = "anomaly"
            
            # Apply injection
            injector = available_injectors[injection_type]
            
            try:
                field_injections_this_row = 0
                if injection_type == "error":
                    # Error injection expects a DataFrame, so create a single-row DF
                    single_row_df = pd.DataFrame([corrupted_df.iloc[row_idx]])
                    modified_df, injections = injector.inject_errors(
                        single_row_df, field_name, max_errors=1, error_probability=1.0
                    )
                    
                    # If error injection failed, try again with higher max_errors to get more attempts
                    max_retries = 3
                    retry_count = 0
                    while not injections and retry_count < max_retries:
                        retry_count += 1
                        single_row_df = pd.DataFrame([corrupted_df.iloc[row_idx]])
                        modified_df, injections = injector.inject_errors(
                            single_row_df, field_name, max_errors=5, error_probability=1.0
                        )
                    
                    if injections:
                        # Use the correct index (the index of the first row in modified_df)
                        modified_row_index = modified_df.index[0]
                        corrupted_value = modified_df.at[modified_row_index, column_name]
                        if corrupted_value != original_value:
                            corrupted_df.at[row_idx, column_name] = corrupted_value
                            
                            # Store injection metadata
                            if field_name not in injection_metadata:
                                injection_metadata[field_name] = []
                            
                            injection_metadata[field_name].append({
                                "row_index": row_idx,
                                "original_value": original_value,
                                "corrupted_value": corrupted_value,
                                "injection_type": injection_type,
                                "rule_name": injections[0].get("error_rule", "unknown"),
                                "message": f"Validation Error: {injections[0].get('error_rule', 'Unknown error')}"
                            })
                            total_injections += 1
                            row_had_injection = True
                            field_injections_this_row += 1
                
                else:  # anomaly injection
                    # Anomaly injection works on individual values
                    single_row_df = pd.DataFrame([corrupted_df.iloc[row_idx]])
                    modified_df, injections = injector.inject_anomalies(
                        single_row_df, field_name, max_anomalies=1, anomaly_probability=1.0, field_mapper=field_mapper
                    )
                    if injections:
                        # Use the correct index (the index of the first row in modified_df)
                        modified_row_index = modified_df.index[0]
                        corrupted_value = modified_df.at[modified_row_index, column_name]
                        if corrupted_value != original_value:
                            corrupted_df.at[row_idx, column_name] = corrupted_value
                            
                            # Store injection metadata
                            if field_name not in injection_metadata:
                                injection_metadata[field_name] = []
                            
                            injection_metadata[field_name].append({
                                "row_index": row_idx,
                                "original_value": original_value,
                                "corrupted_value": corrupted_value,
                                "injection_type": injection_type,
                                "rule_name": injections[0].get("rule_name", "unknown"),
                                "message": f"Semantic Anomaly: {injections[0].get('rule_name', 'Unknown anomaly')}"
                            })
                            total_injections += 1
                            row_had_injection = True
                            field_injections_this_row += 1
            
            except Exception as e:
                print(f"   ⚠️  Injection failed for {field_name} at row {row_idx}: {e}")
        
        if row_had_injection:
            affected_rows += 1
    
    print(f"✅ Comprehensive sample generated:")
    print(f"   📊 Total injections: {total_injections}")
    print(f"   🎯 Affected rows: {affected_rows} / {len(corrupted_df)} ({affected_rows/len(corrupted_df)*100:.1f}%)")
    
    if injection_metadata:
        print(f"   📋 Fields with injections: {len(injection_metadata)}")
        for field_name, injections in injection_metadata.items():
            error_count = sum(1 for inj in injections if inj["injection_type"] == "error")
            anomaly_count = sum(1 for inj in injections if inj["injection_type"] == "anomaly")
            total_field = error_count + anomaly_count
            if error_count > 0 and anomaly_count > 0:
                print(f"      🔍 {field_name}: {total_field} injections ({error_count} errors, {anomaly_count} anomalies)")
            elif error_count > 0:
                print(f"      📝 {field_name}: {error_count} errors")
            else:
                print(f"      🔍 {field_name}: {anomaly_count} anomalies")
    else:
        print(f"   ⚠️  No injections were successful")
    
    return corrupted_df, injection_metadata


def save_comprehensive_sample(sample_df: pd.DataFrame, 
                            injection_metadata: Dict[str, List[Dict[str, Any]]],
                            output_dir: str,
                            sample_name: str = "comprehensive_sample") -> Dict[str, str]:
    """
    Save the comprehensive sample and metadata to files.
    
    Returns:
        Dict with file paths that were created
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the corrupted sample data
    sample_path = os.path.join(os.path.dirname(__file__), output_dir, f"{sample_name}.csv")
    sample_df.to_csv(sample_path, index=False)
    # Do not save metadata or summary files anymore
    return {
        "sample_csv": sample_path
    } 