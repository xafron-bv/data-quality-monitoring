"""
Error Injection Module

This module provides utilities for injecting errors into data based on rule definitions.
It contains the core error injection logic used by both the evaluation system and
the ML training system for data quality monitoring.

The module supports various error injection operations including:
- String replacement
- Regex replacement  
- Text prepending/appending
- Random noise injection
- Conditional error application

Usage:
    from error_injection import apply_error_rule, generate_error_samples, ErrorInjector
    
    # Apply a single error rule
    corrupted_text = apply_error_rule("original text", rule_dict)
    
    # Generate multiple error samples
    samples = generate_error_samples(df, "field_name", rules, num_samples=10)
    
    # Use the ErrorInjector class for more control
    injector = ErrorInjector(rules)
    corrupted_data = injector.inject_errors(df, "field_name", max_errors=3)
"""

import pandas as pd
import random
import os
import json
import string
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from field_mapper import FieldMapper
from exceptions import FileOperationError, ConfigurationError


class ErrorInjector:
    """
    A class for injecting errors into data based on rule definitions.
    
    This class provides a high-level interface for error injection operations,
    allowing for consistent error generation across different components of the
    data quality monitoring system.
    """
    
    def __init__(self, rules: List[Dict[str, Any]], field_mapper: Optional[FieldMapper] = None):
        """
        Initialize the ErrorInjector with a list of error rules.
        
        Args:
            rules: List of error rule dictionaries
            field_mapper: Optional field mapper service (uses default if not provided)
        """
        self.rules = rules
        self.field_mapper = field_mapper
        if self.field_mapper is None:
            raise ValueError("field_mapper must be provided")
        self._validate_rules()
    
    def _validate_rules(self):
        """Validate that all rules have required fields."""
        for i, rule in enumerate(self.rules):
            if 'operation' not in rule:
                raise ValueError(f"Rule {i} missing required 'operation' field")
    
    def inject_errors(self, 
                     df: pd.DataFrame, 
                     field_name: str,
                     max_errors: int,
                     error_probability: float) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Inject errors into the dataframe based on the rules.
        
        Args:
            df: Source dataframe
            field_name: Field name to inject errors into (will be mapped to column name for CSV access)
            max_errors: Maximum number of errors to inject
            error_probability: Probability of injecting errors in each row
            
        Returns:
            Tuple of (modified dataframe, list of injected errors)
        """
        # Get the column name using centralized field mapping
        column_name = self.field_mapper.validate_column_exists(df, field_name)
        
        df_copy = df.copy()
        injected_errors = []
        
        # Get eligible rows (non-null values)
        eligible_rows = df_copy[df_copy[column_name].notna()].index.tolist()
        
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
                original_data = df_copy.at[idx, column_name]
                
                new_data = apply_error_rule(original_data, rule)
                
                if new_data != original_data:
                    df_copy.at[idx, column_name] = new_data
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
    
    elif op == "digit_replacement":
        # Replace random digits with other digits
        if not data_string:
            return data_string
        
        pattern = params.get("digit_pattern", r"[0-9]")
        replacement_digits = params.get("replacement_digits", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        
        # Find all digit matches
        matches = list(re.finditer(pattern, data_string))
        if not matches:
            return data_string
        
        # Replace a random digit
        match = random.choice(matches)
        original_digit = match.group()
        # Choose a different digit
        available_digits = [d for d in replacement_digits if d != original_digit]
        if not available_digits:
            return data_string
        
        new_digit = random.choice(available_digits)
        return data_string[:match.start()] + new_digit + data_string[match.end():]
    
    elif op == "digit_removal":
        # Remove a random digit
        if not data_string:
            return data_string
        
        position = params.get("position", "random")
        digits = [i for i, char in enumerate(data_string) if char.isdigit()]
        
        if not digits:
            return data_string
        
        if position == "random":
            remove_pos = random.choice(digits)
        else:
            remove_pos = position
        
        return data_string[:remove_pos] + data_string[remove_pos + 1:]
    
    elif op == "digit_duplication":
        # Duplicate a random digit
        if not data_string:
            return data_string
        
        position = params.get("position", "random")
        digits = [i for i, char in enumerate(data_string) if char.isdigit()]
        
        if not digits:
            return data_string
        
        if position == "random":
            dup_pos = random.choice(digits)
        else:
            dup_pos = position
        
        char_to_dup = data_string[dup_pos]
        return data_string[:dup_pos] + char_to_dup + data_string[dup_pos:]
    
    elif op == "character_substitution":
        # Replace characters with visually similar ones
        if not data_string:
            return data_string
        
        substitutions = params.get("substitutions", {
            "0": "O", "1": "l", "5": "S", "6": "G", "8": "B"
        })
        
        result = data_string
        for original, replacement in substitutions.items():
            if original in result:
                result = result.replace(original, replacement, 1)  # Replace only first occurrence
                break
        
        return result
    
    elif op == "value_replacement":
        # Replace entire value with a random value from the replacement list (used by anomaly injection)
        replacement_values = params.get("replacement_values", [])
        if replacement_values:
            return random.choice(replacement_values)
        else:
            return data_string
    
    # Unknown operation - return original data
    return data_string


def generate_error_samples(df: pd.DataFrame, 
                         field_name: str, 
                         rules: List[Dict[str, Any]], 
                         num_samples: int,
                         max_errors_per_sample: int,
                         error_probability: float,
                         output_dir: Optional[str] = None,
                         field_mapper: Optional[FieldMapper] = None) -> List[Dict[str, Any]]:
    """
    Generate multiple samples with errors injected based on the rules.
    
    Args:
        df: Source dataframe
        field_name: Field name to inject errors into (will be mapped to column name for CSV access)
        rules: List of error rules
        num_samples: Number of samples to generate
        max_errors_per_sample: Maximum number of errors per sample
        output_dir: Optional directory to save samples (if None, samples are not saved)
        field_mapper: Optional field mapper service (uses default if not provided)
        error_probability: Probability of injecting errors in each row
        
    Returns:
        List of sample dictionaries containing 'data' and 'injected_errors'
    """
    if not rules:
        raise ValueError("No error rules provided")
    
    if field_mapper is None:
        # This function needs a field_mapper to be passed
        raise ValueError("This function requires a field_mapper parameter")
    
    # Validate that the column exists in the dataframe
    column_name = field_mapper.validate_column_exists(df, field_name)
    
    samples = []
    injector = ErrorInjector(rules, field_mapper)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {num_samples} samples in '{output_dir}' with up to {max_errors_per_sample} errors each...")
    else:
        print(f"Generating {num_samples} samples with up to {max_errors_per_sample} errors each...")

    for i in range(num_samples):
        # Generate errors for this sample
        num_to_inject = random.randint(1, max_errors_per_sample)
        df_with_errors, injected_errors = injector.inject_errors(
            df, field_name, max_errors=num_to_inject, error_probability=error_probability
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
        FileOperationError: If the rules file cannot be loaded or parsed
        ConfigurationError: If the rules file format is invalid
    """
    
    try:
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
    except FileNotFoundError:
        raise FileOperationError(
            f"Rules file not found: {rules_file_path}",
            details={
                'file_path': rules_file_path,
                'suggestion': 'Ensure the error injection rules file exists at the specified path'
            }
        )
    except json.JSONDecodeError as e:
        raise FileOperationError(
            f"Invalid JSON in rules file: {rules_file_path}",
            details={
                'file_path': rules_file_path,
                'json_error': str(e),
                'suggestion': 'Check the JSON syntax in the rules file'
            }
        ) from e
    
    # Extract rules from the standard format
    if isinstance(rules_data, dict) and "error_rules" in rules_data:
        rules = rules_data["error_rules"]
    elif isinstance(rules_data, list):
        rules = rules_data
    else:
        raise ConfigurationError(
            "Rules file must contain 'error_rules' key or be a list of rules",
            details={
                'file_path': rules_file_path,
                'found_type': type(rules_data).__name__,
                'found_keys': list(rules_data.keys()) if isinstance(rules_data, dict) else None
            }
        )
    
    # Add rule names if not present
    for i, rule in enumerate(rules):
        if "rule_name" not in rule:
            rule["rule_name"] = f"{rule.get('operation', 'unknown')}_{i}"
    
    return rules


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
