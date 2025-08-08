#!/usr/bin/env python3
"""
Generic Pattern-Based Anomaly Detector

This module provides a unified pattern-based anomaly detection system that loads
field-specific rules from JSON configuration files rather than hardcoded logic.
"""

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class PatternBasedDetector(AnomalyDetectorInterface):
    """
    Generic pattern-based anomaly detector that loads field-specific rules from JSON files.

    Rules files should be located in: anomaly_detectors/pattern_based/rules/{field_name}/{variation}.json
    """

    class ErrorCode(str, Enum):
        """Generic error codes for pattern-based detection."""
        INVALID_VALUE = "INVALID_VALUE"
        UNKNOWN_VALUE = "UNKNOWN_VALUE"
        INVALID_FORMAT = "INVALID_FORMAT"
        SUSPICIOUS_PATTERN = "SUSPICIOUS_PATTERN"
        DOMAIN_VIOLATION = "DOMAIN_VIOLATION"

    def __init__(self, field_name: str, variation: Optional[str] = None):
        """
        Initialize the detector for a specific field.

        Args:
            field_name: The name of the field to detect anomalies for
            variation: Optional variation key for brand/format-specific rules
        """
        self.field_name = field_name
        self.variation = variation
        self.rules = {}
        self.known_values = set()
        self.format_patterns = []
        self.validation_rules = []

        # Load field-specific rules
        self._load_rules()

    def _load_rules(self):
        """Load field-specific rules from JSON file."""
        rules_dir = Path(__file__).parent / "rules"

        # Require variation
        if not self.variation:
            raise ValueError(f"Variation is required for field '{self.field_name}'")

        rules_file = rules_dir / self.field_name / f"{self.variation}.json"
        if not rules_file.exists():
            raise FileNotFoundError(
                f"Pattern rules for field '{self.field_name}' variation '{self.variation}' not found at {rules_file}"
            )

        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)

            self._extract_components_from_rules()

        except Exception as e:
            print(f"Warning: Could not load rules for field '{self.field_name}': {e}")
            self.rules = {}

    def _extract_components_from_rules(self):
        """Extract rule components from self.rules"""
        # Filter out comments and empty strings from known_values
        raw_values = self.rules.get('known_values', [])
        self.known_values = set(
            val.lower() for val in raw_values 
            if val and not str(val).strip().startswith('#')
        )
        self.format_patterns = self.rules.get('format_patterns', [])
        self.validation_rules = self.rules.get('validation_rules', [])

    def _create_default_rules_file(self, rules_file: Path):
        """Create a default rules file template."""
        rules_file.parent.mkdir(parents=True, exist_ok=True)

        default_rules = {
            "field_name": self.field_name,
            "description": f"Pattern-based rules for {self.field_name} field",
            "known_values": [
                "# Add known valid values here (case-insensitive)",
                "# Example: valid_option_1, valid_option_2"
            ],
            "format_patterns": [
                {
                    "name": "basic_format",
                    "description": "Basic format validation",
                    "pattern": "^[a-zA-Z0-9\\s\\-_%/]+$",
                    "message": "Contains invalid characters"
                }
            ],
            "validation_rules": [
                {
                    "name": "not_empty",
                    "description": "Value should not be empty",
                    "type": "not_empty",
                    "message": "Value cannot be empty"
                },
                {
                    "name": "max_length",
                    "description": "Value should not be too long",
                    "type": "max_length",
                    "max_length": 100,
                    "message": "Value is too long"
                }
            ]
        }

        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(default_rules, f, indent=2, ensure_ascii=False)

    def _normalize_value(self, value: str) -> str:
        """Normalize a value for comparison."""
        if not isinstance(value, str):
            return ""
        return value.lower().strip()

    def _check_format_patterns(self, value: str) -> Optional[AnomalyError]:
        """Check value against format patterns."""
        for pattern_rule in self.format_patterns:
            pattern = pattern_rule.get('pattern', '')
            if pattern and not re.match(pattern, value):
                return AnomalyError(
                    anomaly_type=self.ErrorCode.INVALID_FORMAT,
                    probability=pattern_rule.get('probability', 0.8),
                    details={
                        "field": self.field_name,
                        "value": value,
                        "rule": pattern_rule.get('name', 'unknown'),
                        "message": pattern_rule.get('message', 'Invalid format'),
                        "pattern": pattern
                    }
                )
        return None

    def _check_validation_rules(self, value: str) -> Optional[AnomalyError]:
        """Check value against validation rules."""
        for validation_rule in self.validation_rules:
            rule_type = validation_rule.get('type', '')

            if rule_type == 'not_empty' and not value.strip():
                return AnomalyError(
                    anomaly_type=self.ErrorCode.INVALID_VALUE,
                    probability=validation_rule.get('probability', 0.9),
                    details={
                        "field": self.field_name,
                        "value": value,
                        "rule": validation_rule.get('name', 'not_empty'),
                        "message": validation_rule.get('message', 'Value cannot be empty')
                    }
                )

            elif rule_type == 'max_length':
                max_len = validation_rule.get('max_length', 100)
                if len(value) > max_len:
                    return AnomalyError(
                        anomaly_type=self.ErrorCode.INVALID_VALUE,
                        probability=validation_rule.get('probability', 0.8),
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": validation_rule.get('name', 'max_length'),
                            "message": validation_rule.get('message', 'Value is too long'),
                            "actual_length": len(value),
                            "max_length": max_len
                        }
                    )

            elif rule_type == 'min_length':
                min_len = validation_rule.get('min_length', 1)
                if len(value) < min_len:
                    return AnomalyError(
                        anomaly_type=self.ErrorCode.INVALID_VALUE,
                        probability=validation_rule.get('probability', 0.8),
                        details={
                            "field": self.field_name,
                            "value": value,
                            "rule": validation_rule.get('name', 'min_length'),
                            "message": validation_rule.get('message', 'Value is too short'),
                            "actual_length": len(value),
                            "min_length": min_len
                        }
                    )

        return None

    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies using loaded rules.

        Args:
            value: The value to check for anomalies
            context: Optional context information

        Returns:
            AnomalyError if anomaly detected, None otherwise
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None

        value_str = str(value).strip()
        normalized_value = self._normalize_value(value_str)

        # Check format patterns first
        format_error = self._check_format_patterns(value_str)
        if format_error:
            return format_error

        # Check validation rules
        validation_error = self._check_validation_rules(value_str)
        if validation_error:
            return validation_error

        # Check against known values list (only if not empty)
        # Skip this check if known_values is empty (no restrictions)
        if self.known_values and len(self.known_values) > 0 and normalized_value not in self.known_values:
            # Look for partial matches to suggest possible typos
            close_matches = [known for known in self.known_values
                           if known.startswith(normalized_value[:3]) or normalized_value.startswith(known[:3])]

            probability = 0.75 if close_matches else 0.85
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNKNOWN_VALUE,
                probability=probability,
                details={
                    "field": self.field_name,
                    "value": value_str,
                    "reason": "Value not in known values list",
                    "possible_matches": close_matches[:3] if close_matches else [],
                    "known_values_count": len(self.known_values)
                }
            )

        # No anomalies detected
        return None

    def get_detector_args(self) -> Dict[str, Any]:
        """Return arguments needed to recreate this detector instance."""
        return {"field_name": self.field_name}


def create_detector(field_name: str) -> PatternBasedDetector:
    """Factory function to create a pattern-based detector for a field."""
    return PatternBasedDetector(field_name)
