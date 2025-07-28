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

import json
import os
import sys
import argparse
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def main():
    parser = argparse.ArgumentParser(description="Generate detection weights from performance results")
    parser.add_argument("--input-file", "-i", required=True,
                       help="Path to unified report JSON file with performance data")
        parser.add_argument("--output-file", "-o", default=None,
                        help="Output file for generated weights (default: ./detection_weights.json)")
    parser.add_argument("--baseline-weight", "-b", type=float, default=0.1,
                       help="Baseline weight for untrained methods (default: 0.1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed weight information")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found")
        return 1
    
    try:
        print(f"📊 Loading performance data from: {args.input_file}")
        performance_data = load_performance_data(args.input_file)
        
        if not performance_data:
            print(f"❌ Error: No field performance data found in {args.input_file}")
            return 1
        
        print(f"✅ Found performance data for {len(performance_data)} fields")
        
        # Generate weights
        print(f"🔧 Generating weights with baseline weight: {args.baseline_weight}")
        field_weights = generate_weights_from_performance(performance_data, args.baseline_weight)
        
        # Create comprehensive report
        weights_report = generate_weights_report(field_weights, performance_data, args.input_file)
        
        # Set default output file if not provided
        output_file = args.output_file if args.output_file else os.path.join(
            os.path.dirname(__file__), "generate_detection_weights", "reports", "detection_weights.json"
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save weights report
        print(f"💾 Saving weights to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(weights_report, f, indent=2)
        
        # Print summary if verbose
        if args.verbose:
            print(f"\n📋 Generated Weights Summary:")
            for field_name, summary in weights_report["weight_summary"].items():
                weights_str = ", ".join([f"{method}: {weight:.2f}" 
                                       for method, weight in summary["weights"].items()])
                print(f"   {field_name}: {weights_str}")
                print(f"      → Dominant: {summary['dominant_method']} ({summary['dominant_weight']:.2f})")
            
            print(f"\n🔍 Performance Insights:")
            for field_name, insights in weights_report["performance_insights"].items():
                print(f"   {field_name}:")
                for insight in insights:
                    print(f"      • {insight}")
        
        print(f"\n✅ Detection weights generated successfully!")
        print(f"📁 Output file: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating weights: {e}")
        return 1


if __name__ == "__main__":
    exit(main())