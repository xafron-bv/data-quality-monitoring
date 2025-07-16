"""
Error Injection Module

This module provides utilities for injecting errors into data based on rule definitions.
It contains the core error injection logic used by both the evaluation system and
the ML training system for data quality monitoring.

The module supports various error injection operations including:
- String replacement
- Regex replacement  
- Text prepending/appending
- Whitespace addition
- Random noise injection
- Conditional error application

Usage:
    from error_injection import apply_error_rule, generate_error_samples, ErrorInjector
    
    # Apply a single error rule
    corrupted_text = apply_error_rule("original text", rule_dict)
    
    # Generate multiple error samples
    samples = generate_error_samples(df, "column_name", rules, num_samples=10)
    
    # Use the ErrorInjector class for more control
    injector = ErrorInjector(rules)
    corrupted_data = injector.inject_errors(df, "column_name", max_errors=3)
"""

import pandas as pd
import numpy as np
import random
import os
import json
import string
import re
from typing import List, Dict, Any, Union, Optional, Tuple


class ErrorInjector:
    """
    A class for injecting errors into data based on rule definitions.
    
    This class provides a high-level interface for error injection operations,
    allowing for consistent error generation across different components of the
    data quality monitoring system.
    """
    
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize the ErrorInjector with a list of error rules.
        
        Args:
            rules: List of error rule dictionaries
        """
        self.rules = rules
        self._validate_rules()
    
    def _validate_rules(self):
        """Validate that all rules have required fields."""
        for i, rule in enumerate(self.rules):
            if 'operation' not in rule:
                raise ValueError(f"Rule {i} missing required 'operation' field")
    
    def inject_errors(self, 
                     df: pd.DataFrame, 
                     column: str, 
                     max_errors: int = 3,
                     error_probability: float = 0.1) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Inject errors into a dataframe column.
        
        Args:
            df: The dataframe to inject errors into
            column: The column name to inject errors into
            max_errors: Maximum number of errors to inject
            error_probability: Probability of injecting an error in each row
            
        Returns:
            Tuple of (modified_dataframe, list_of_injected_errors)
        """
        df_copy = df.copy()
        injected_errors = []
        
        # Get eligible rows (non-null values)
        eligible_rows = df_copy[df_copy[column].notna()].index.tolist()
        
        if not eligible_rows:
            return df_copy, injected_errors
        
        # Determine how many errors to inject
        num_errors = min(max_errors, len(eligible_rows))
        
        # Apply probability-based selection
        if error_probability < 1.0:
            eligible_rows = [idx for idx in eligible_rows if random.random() < error_probability]
            num_errors = min(num_errors, len(eligible_rows))
        
        # Select random rows to inject errors
        if num_errors > 0:
            error_rows = random.sample(eligible_rows, num_errors)
            
            for idx in error_rows:
                rule = random.choice(self.rules)
                original_data = df_copy.at[idx, column]
                
                new_data = apply_error_rule(original_data, rule)
                
                if new_data != original_data:
                    df_copy.at[idx, column] = new_data
                    injected_errors.append({
                        "row_index": idx,
                        "original_data": original_data,
                        "injected_data": new_data,
                        "error_rule": rule.get('rule_name', rule.get('operation', 'unknown')),
                        "rule_operation": rule['operation']
                    })
        
        return df_copy, injected_errors
    
    def generate_corrupted_text(self, text: str, num_corruptions: int = 1) -> str:
        """
        Generate a corrupted version of the input text.
        
        Args:
            text: The original text to corrupt
            num_corruptions: Number of corruption operations to apply
            
        Returns:
            Corrupted version of the text
        """
        corrupted = text
        for _ in range(num_corruptions):
            rule = random.choice(self.rules)
            corrupted = apply_error_rule(corrupted, rule)
        return corrupted


def apply_error_rule(data_string: Union[str, Any], rule: Dict[str, Any]) -> Union[str, Any]:
    """
    Apply a single error rule to a string.
    
    This function applies various types of error injection operations based on the
    rule definition. It supports conditional application of rules and multiple
    operation types.
    
    Args:
        data_string: The input data to apply the rule to
        rule: Dictionary containing the rule definition
        
    Returns:
        The modified data after applying the rule
        
    Supported Operations:
        - string_replace: Replace all occurrences of a substring
        - regex_replace: Replace using regular expressions
        - prepend: Add text to the beginning
        - append: Add text to the end
        - add_whitespace: Add whitespace around the text
        - random_noise: Add random characters or duplicate words
        - regex_extract_validate: Extract and validate using regex
        - replace_with: Replace entire text with specified value
    """
    if not isinstance(data_string, str):
        return data_string

    # Check conditions first
    if "conditions" in rule and rule["conditions"]:
        condition_met = False
        for cond in rule["conditions"]:
            if cond["type"] == "contains":
                if str(cond["value"]) in str(data_string):
                    condition_met = True
                    break
            elif cond["type"] == "not_contains":
                if str(cond["value"]) not in str(data_string):
                    condition_met = True
                    break
            elif cond["type"] == "equals":
                if str(data_string) == str(cond["value"]):
                    condition_met = True
                    break
            elif cond["type"] == "regex_match":
                if re.search(cond["pattern"], str(data_string)):
                    condition_met = True
                    break
        
        if not condition_met:
            return data_string

    # Apply probability check if specified
    if "probability" in rule:
        if random.random() > rule["probability"]:
            return data_string

    op = rule["operation"]
    params = rule.get("params", {})

    if op == "string_replace":
        find_str = str(params["find"])
        replace_str = str(params["replace"])
        return data_string.replace(find_str, replace_str)
    
    elif op == "regex_replace":
        pattern = params["pattern"]
        replace_str = params["replace"]
        count = params.get("count", 0)  # Replace all if count is 0
        return re.sub(pattern, replace_str, data_string, count=count)
    
    elif op == "prepend":
        return params["text"] + data_string
    
    elif op == "append":
        return data_string + params["text"]
    
    elif op == "add_whitespace":
        return " " + data_string + " "
    
    elif op == "replace_with":
        return params["text"]
    
    elif op == "random_noise":
        if not data_string:
            return data_string
            
        noise_type = params.get("type", "chars")
        
        if noise_type == "chars":
            # Add random characters
            noise_chars = params.get("chars", "!@#$%^&*()[]{}|;:\",./<>?")
            noise_length = params.get("length", 1)
            noise = ''.join(random.choices(noise_chars, k=noise_length))
            
            # Insert at random position
            pos = random.randint(0, len(data_string))
            return data_string[:pos] + noise + data_string[pos:]
        
        elif noise_type == "duplicate":
            # Duplicate a random word
            parts = data_string.split()
            if not parts:
                return data_string
            return data_string + " " + random.choice(parts)
        
        elif noise_type == "random_chars":
            # Add random alphanumeric characters
            noise_length = params.get("length", 4)
            noise = ''.join(random.choices(string.ascii_letters + string.digits, k=noise_length))
            pos = random.randint(0, len(data_string))
            return data_string[:pos] + noise + data_string[pos:]
        
        else:
            # Default: add random special character
            char = random.choice('!@#$%^&*()[]{}|;:",./<>?')
            pos = random.randint(0, len(data_string))
            return data_string[:pos] + char + data_string[pos:]
    
    elif op == "regex_extract_validate":
        extract_pattern = params["extract_pattern"]
        match = re.search(extract_pattern, data_string)
        if match and match.lastindex is not None and match.lastindex >= 1:
            extracted_value = match.group(1)
            # Evaluate the validation condition
            try:
                if eval(params["validation"], {"value": extracted_value}):
                    return extracted_value
                else:
                    return data_string
            except Exception as e:
                print(f"Validation expression error: {e}")
                return data_string
        else:
            return data_string
    
    elif op == "case_change":
        change_type = params.get("type", "upper")
        if change_type == "upper":
            return data_string.upper()
        elif change_type == "lower":
            return data_string.lower()
        elif change_type == "title":
            return data_string.title()
        elif change_type == "swap":
            return data_string.swapcase()
    
    elif op == "truncate":
        max_length = params.get("length", len(data_string) // 2)
        return data_string[:max_length]
    
    elif op == "reverse":
        return data_string[::-1]
    
    return data_string


def generate_error_samples(df: pd.DataFrame, 
                         column: str, 
                         rules: List[Dict[str, Any]], 
                         num_samples: int,
                         max_errors_per_sample: int = 3,
                         output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate multiple samples with errors injected based on the rules.
    
    Args:
        df: Source dataframe
        column: Column name to inject errors into
        rules: List of error rules
        num_samples: Number of samples to generate
        max_errors_per_sample: Maximum number of errors per sample
        output_dir: Optional directory to save samples (if None, samples are not saved)
        
    Returns:
        List of sample dictionaries containing 'data' and 'injected_errors'
    """
    if not rules:
        raise ValueError("No error rules provided")
    
    samples = []
    injector = ErrorInjector(rules)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {num_samples} samples in '{output_dir}' with up to {max_errors_per_sample} errors each...")
    else:
        print(f"Generating {num_samples} samples with up to {max_errors_per_sample} errors each...")

    for i in range(num_samples):
        # Generate errors for this sample
        num_to_inject = random.randint(1, max_errors_per_sample)
        df_with_errors, injected_errors = injector.inject_errors(
            df, column, max_errors=num_to_inject
        )
        
        # Save the sample if output directory is specified
        if output_dir:
            sample_csv_path = os.path.join(output_dir, f'sample_{i}.csv')
            injected_errors_path = os.path.join(output_dir, f'sample_{i}_injected_errors.json')
            
            df_with_errors.to_csv(sample_csv_path, index=False)
            with open(injected_errors_path, 'w') as f:
                json.dump(injected_errors, f, indent=4)
        
        samples.append({
            "data": df_with_errors, 
            "injected_errors": injected_errors,
            "sample_index": i
        })
    
    print("Sample generation complete.")
    return samples


def load_error_rules(rules_file_path: str) -> List[Dict[str, Any]]:
    """
    Load error rules from a JSON file.
    
    Args:
        rules_file_path: Path to the JSON file containing error rules
        
    Returns:
        List of error rule dictionaries
        
    Raises:
        FileNotFoundError: If the rules file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        # Extract rules from the standard format
        if isinstance(rules_data, dict) and "error_rules" in rules_data:
            rules = rules_data["error_rules"]
        elif isinstance(rules_data, list):
            rules = rules_data
        else:
            raise ValueError("Rules file must contain 'error_rules' key or be a list of rules")
        
        # Add rule names if not present
        for i, rule in enumerate(rules):
            if "rule_name" not in rule:
                rule["rule_name"] = f"{rule.get('operation', 'unknown')}_{i}"
        
        return rules
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found: {rules_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in rules file: {rules_file_path}", e.doc, e.pos)


def create_corrupted_variants(texts: List[str], 
                            rules: List[Dict[str, Any]], 
                            num_variants_per_text: int = 1) -> List[str]:
    """
    Create corrupted variants of input texts for ML training.
    
    Args:
        texts: List of clean texts
        rules: List of error rules
        num_variants_per_text: Number of corrupted variants to create per text
        
    Returns:
        List of corrupted text variants
    """
    injector = ErrorInjector(rules)
    corrupted_variants = []
    
    for text in texts:
        for _ in range(num_variants_per_text):
            corrupted = injector.generate_corrupted_text(text)
            if corrupted != text:  # Only add if corruption actually occurred
                corrupted_variants.append(corrupted)
    
    return corrupted_variants


# Utility functions for backward compatibility
def apply_error_rules_bulk(df: pd.DataFrame, 
                          column: str, 
                          rules: List[Dict[str, Any]],
                          max_errors: int = 3) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply error rules to a dataframe column (backward compatibility function).
    
    Args:
        df: The dataframe to modify
        column: Column name to inject errors into
        rules: List of error rules
        max_errors: Maximum number of errors to inject
        
    Returns:
        Tuple of (modified_dataframe, injected_errors_list)
    """
    injector = ErrorInjector(rules)
    return injector.inject_errors(df, column, max_errors=max_errors)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_rules = [
        {
            "operation": "string_replace",
            "rule_name": "test_replace",
            "params": {"find": "test", "replace": "tset"}
        },
        {
            "operation": "add_whitespace",
            "rule_name": "add_spaces"
        },
        {
            "operation": "random_noise",
            "rule_name": "add_noise",
            "params": {"type": "chars", "length": 2}
        }
    ]
    
    # Test single rule application
    test_text = "This is a test string"
    corrupted = apply_error_rule(test_text, sample_rules[0])
    print(f"Original: {test_text}")
    print(f"Corrupted: {corrupted}")
    
    # Test ErrorInjector
    injector = ErrorInjector(sample_rules)
    corrupted_variant = injector.generate_corrupted_text(test_text, num_corruptions=2)
    print(f"Multi-corrupted: {corrupted_variant}")
