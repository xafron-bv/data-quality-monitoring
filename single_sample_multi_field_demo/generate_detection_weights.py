#!/usr/bin/env python3
"""
Detection Weight Generator

This tool analyzes performance results from demo outputs and generates
field-specific weights for anomaly detection methods based on their
F1 scores, precision, and recall metrics.

The generated weights are used by the weighted combination detection system
to dynamically prioritize the most effective detection method for each field,
improving overall detection accuracy compared to the fixed priority approach.

Usage:
    python3 generate_detection_weights.py -i demo_results/report.json -o weights.json --verbose

Created as part of the weighted combination detection enhancement.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def load_performance_data(results_file: str) -> Dict[str, Any]:
    """Load performance data from a unified report JSON file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data.get('fields', {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading performance data from {results_file}: {e}")


def generate_weights_from_performance(performance_data: Dict[str, Any],
                                    baseline_weight: float = 0.1) -> Dict[str, Dict[str, float]]:
    """
    Generate detection weights based on performance metrics.

    Args:
        performance_data: Performance data by field
        baseline_weight: Minimum weight for untrained/poor performing methods

    Returns:
        Dict mapping field_name -> method -> weight
    """
    detection_methods = ["pattern_based", "ml_based", "llm_based"]
    field_weights = {}

    for field_name, field_data in performance_data.items():
        field_weights[field_name] = {}
        method_scores = {}

        # Calculate F1 scores for each detection method
        for method in detection_methods:
            if method in field_data:
                method_data = field_data[method]
                precision = method_data.get('precision', 0.0)
                recall = method_data.get('recall', 0.0)
                f1_score = calculate_f1_score(precision, recall)
                method_scores[method] = f1_score
            else:
                method_scores[method] = 0.0

        # Calculate total F1 score for normalization
        total_f1 = sum(method_scores.values())

        # Assign weights based on performance
        if total_f1 == 0:
            # If no method has performance, give equal weights
            num_methods = len(detection_methods)
            for method in detection_methods:
                field_weights[field_name][method] = 1.0 / num_methods
        else:
            # Weight methods by their F1 score performance with baseline
            for method in detection_methods:
                f1_score = method_scores[method]
                field_weights[field_name][method] = max(f1_score, baseline_weight)

            # Normalize weights to sum to 1
            total_weight = sum(field_weights[field_name].values())
            for method in detection_methods:
                field_weights[field_name][method] /= total_weight

    return field_weights


def generate_weights_report(field_weights: Dict[str, Dict[str, float]],
                          performance_data: Dict[str, Any],
                          source_file: str) -> Dict[str, Any]:
    """Generate a comprehensive weights report with metadata."""

    # Calculate summary statistics
    weight_summary = {}
    for field_name, weights in field_weights.items():
        dominant_method = max(weights.items(), key=lambda x: x[1])
        weight_summary[field_name] = {
            "dominant_method": dominant_method[0],
            "dominant_weight": round(dominant_method[1], 3),
            "weights": {method: round(weight, 3) for method, weight in weights.items()}
        }

    # Create performance insights
    performance_insights = {}
    for field_name, field_data in performance_data.items():
        insights = []
        for method in ["pattern_based", "ml_based", "llm_based"]:
            if method in field_data:
                method_data = field_data[method]
                precision = method_data.get('precision', 0.0)
                recall = method_data.get('recall', 0.0)
                f1 = calculate_f1_score(precision, recall)

                if f1 > 0.8:
                    insights.append(f"{method}: Excellent (F1={f1:.3f})")
                elif f1 > 0.5:
                    insights.append(f"{method}: Good (F1={f1:.3f})")
                elif f1 > 0.1:
                    insights.append(f"{method}: Poor (F1={f1:.3f})")
                else:
                    insights.append(f"{method}: Not effective (F1={f1:.3f})")
            else:
                insights.append(f"{method}: Not trained/available")

        performance_insights[field_name] = insights

    return {
        "metadata": {
            "description": "Field-specific weights for anomaly detection methods",
            "source_file": source_file,
            "calculation_method": "F1-score based with baseline normalization",
            "baseline_weight": 0.1,
            "generated_by": "generate_detection_weights.py"
        },
        "weights": field_weights,
        "weight_summary": weight_summary,
        "performance_insights": performance_insights
    }
