"""
JSON-based validator that reads validation rules from JSON files.
Similar to pattern-based anomaly detectors but for validation.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface


class JSONValidator(ValidatorInterface):
    """
    A generic validator that reads validation rules from JSON files.
    This replaces individual Python validation files with a unified JSON-based approach.
    """
    
    def __init__(self, field_name: str, variation: Optional[str] = None):
        """
        Initialize the validator for a specific field.
        
        Args:
            field_name: The name of the field to validate (e.g., 'category', 'size')
            variation: Optional variation key for brand/format-specific rules
        """
        self.field_name = field_name
        self.variation = variation
        self.rules = self._load_validation_rules()
        self.error_messages = self.rules.get('error_messages', {})
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from the JSON file."""
        base_rules_dir = os.path.join(
            os.path.dirname(__file__),
            'rules'
        )

        # Enforce variation is specified
        if not self.variation:
            raise ValueError(f"Variation is required for field '{self.field_name}'")

        # Only allow variation-specific rules under rules/{field}/{variation}.json
        field_dir = os.path.join(base_rules_dir, f'{self.field_name}')
        variant_rules_path = os.path.join(field_dir, f'{self.variation}.json')
        if not os.path.exists(variant_rules_path):
            raise FileNotFoundError(
                f"Validation rules for field '{self.field_name}' variation '{self.variation}' not found at {variant_rules_path}"
            )
        with open(variant_rules_path, 'r') as f:
            return json.load(f)
    
    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """
        Validate a single data entry against all rules defined in the JSON file.
        
        Args:
            value: The data to be validated
            
        Returns:
            ValidationError if validation fails, None if valid
        """
        # Run through each validation rule
        for rule in self.rules.get('validation_rules', []):
            error = self._apply_rule(value, rule)
            if error:
                return error
        
        return None
    
    def _apply_rule(self, value: Any, rule: Dict[str, Any]) -> Optional[ValidationError]:
        """Apply a single validation rule to a value."""
        rule_type = rule.get('type')
        error_code = rule.get('error_code')
        probability = rule.get('probability', 1.0)
        
        # Check for missing values
        if rule_type == 'missing':
            if pd.isna(value):
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={}
                )
                
        # Skip other checks if value is missing
        if pd.isna(value):
            return None
            
        # Type checking
        if rule_type == 'type_check':
            expected_type = rule.get('expected_type')
            if expected_type == 'string' and not isinstance(value, str):
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={"expected": expected_type, "actual": str(type(value).__name__)}
                )
                
        # Convert to string for remaining checks
        value_str = str(value) if not isinstance(value, str) else value
        
        # Empty string check
        if rule_type == 'empty_string':
            if value_str == "":
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={}
                )
                
        # Whitespace check
        if rule_type == 'whitespace':
            if value_str.strip() != value_str:
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={"original": value_str, "stripped": value_str.strip()}
                )
                
        # Regex pattern matching
        if rule_type == 'regex':
            pattern = rule.get('pattern')
            case_insensitive = rule.get('case_insensitive', False)
            flags = re.IGNORECASE if case_insensitive else 0
            
            if re.search(pattern, value_str, flags):
                # Check for exclusions
                exclude_patterns = rule.get('exclude_patterns', [])
                for exclude in exclude_patterns:
                    if re.match(exclude, value_str, flags):
                        return None
                        
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={self._get_value_key(): value_str}
                )
                
        # Multiple regex patterns (OR logic)
        if rule_type == 'regex_multiple':
            patterns = rule.get('patterns', [])
            for pattern in patterns:
                if re.search(pattern, value_str):
                    return ValidationError(
                        error_type=error_code,
                        probability=probability,
                        details={self._get_value_key(): value_str}
                    )
                    
        # Negative regex (should NOT match any pattern)
        if rule_type == 'regex_negative':
            patterns = rule.get('patterns', [])
            case_insensitive = rule.get('case_insensitive', False)
            flags = re.IGNORECASE if case_insensitive else 0
            
            matched = False
            for pattern in patterns:
                if re.match(pattern, value_str, flags):
                    matched = True
                    break
                    
            if not matched:
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={self._get_value_key(): value_str}
                )
                
        # Minimum length check
        if rule_type == 'min_length':
            min_length = rule.get('min_length', 0)
            if len(value_str) < min_length:
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={"instructions": value_str}
                )
                
        # Keyword check
        if rule_type == 'keyword_check':
            keywords = rule.get('required_keywords', [])
            case_insensitive = rule.get('case_insensitive', False)
            
            found = False
            test_str = value_str.lower() if case_insensitive else value_str
            for keyword in keywords:
                test_keyword = keyword.lower() if case_insensitive else keyword
                if test_keyword in test_str:
                    found = True
                    break
                    
            if not found:
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={}
                )
                
        # Percentage sum check for materials
        if rule_type == 'percentage_sum_check':
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', value_str)
            if percentages:
                total = sum(float(p) for p in percentages)
                if abs(total - 100) > 0.1:  # Allow small rounding errors
                    return ValidationError(
                        error_type=error_code,
                        probability=probability,
                        details={
                            self._get_value_key(): value_str,
                            "sum": total
                        }
                    )
                    
        # Parenthesis check
        if rule_type == 'parenthesis_check':
            open_count = value_str.count('(')
            close_count = value_str.count(')')
            if open_count != close_count:
                return ValidationError(
                    error_type=error_code,
                    probability=probability,
                    details={self._get_value_key(): value_str}
                )
                
        # Year range check
        if rule_type == 'year_range_check':
            min_year = rule.get('min_year', 1900)
            max_year = rule.get('max_year', 2100)
            
            # Find years in the string
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', value_str)
            for year in years:
                year_int = int(year)
                if year_int < min_year or year_int > max_year:
                    return ValidationError(
                        error_type=error_code,
                        probability=probability,
                        details={
                            self._get_value_key(): value_str,
                            "min_year": min_year,
                            "max_year": max_year
                        }
                    )
                    
        # Temperature check
        if rule_type == 'temperature_check':
            min_celsius = rule.get('min_celsius', -273)
            max_celsius = rule.get('max_celsius', 1000)
            
            # Find temperature values
            temps = re.findall(r'(\d+)\s*°?[CcFf]', value_str)
            for temp in temps:
                temp_val = int(temp)
                # Simple check - assume C if under 100, F if over
                if temp_val > 100:
                    # Convert F to C
                    temp_val = (temp_val - 32) * 5/9
                    
                if temp_val < min_celsius or temp_val > max_celsius:
                    return ValidationError(
                        error_type=error_code,
                        probability=probability,
                        details={"instructions": value_str}
                    )
                    
        # Contradiction check
        if rule_type == 'contradiction_check':
            pairs = rule.get('contradiction_pairs', [])
            lower_value = value_str.lower()
            
            for pair in pairs:
                if len(pair) == 2:
                    if pair[0].lower() in lower_value and pair[1].lower() in lower_value:
                        return ValidationError(
                            error_type=error_code,
                            probability=probability,
                            details={"instructions": value_str}
                        )
        
        return None
    
    def _get_value_key(self) -> str:
        """Get the appropriate key name for the value in error details."""
        # Map field names to their value keys used in error messages
        field_to_key = {
            'category': 'category',
            'color_name': 'color',
            'size': 'size',
            'material': 'material',
            'season': 'season',
            'care_instructions': 'instructions'
        }
        return field_to_key.get(self.field_name, 'value')