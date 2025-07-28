#!/usr/bin/env python3
"""
Detection Curve Generator

This script generates precision-recall and ROC curves for both ML-based and LLM-based
anomaly detection across different thresholds for each field. It uses the existing
evaluation framework to test multiple thresholds and plot the results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from anomaly_detectors.llm_based.llm_anomaly_detector import LLMAnomalyDetector
from anomaly_detectors.ml_based.check_anomalies import check_anomalies, load_model_for_field
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
from anomaly_detectors.ml_based.model_training import preprocess_text
from common.anomaly_injection import load_anomaly_rules
from common.brand_config import get_available_brands, load_brand_config
from common.error_injection import apply_error_rule, load_error_rules
from common.field_mapper import FieldMapper


class DetectionCurveGenerator:
    """Generates precision-recall and ROC curves for ML-based and LLM-based anomaly detection."""

    def __init__(self, data_file: str, output_dir: str = "ml_curves"):
        """
        Initialize the curve generator.

        Args:
            data_file: Path to the CSV data file
            output_dir: Directory to save the generated curves
        """
        self.data_file = data_file
        self.output_dir = output_dir
        # field_mapper will be set when set_brand is called
        self.field_mapper = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        self.df = pd.read_csv(data_file)
        print(f"📊 Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def get_available_ml_fields(self) -> List[str]:
        """Get fields that have trained ML models available."""
        available_fields = []

        # Set the correct results directory path
        results_dir = os.path.join("anomaly_detectors", "results")

        for field_name in self.field_mapper.get_available_fields():
            try:
                # Try to load the model for this field with the correct results directory
                model, column_name, reference_centroid = load_model_for_field(field_name, results_dir=results_dir)
                if column_name in self.df.columns:
                    available_fields.append(field_name)
                    print(f"   ✅ {field_name}: ML model available")
                else:
                    print(f"   ⚠️  {field_name}: ML model available but column '{column_name}' not in dataset")
            except Exception as e:
                print(f"   ❌ {field_name}: No ML model available ({str(e)[:50]}...)")

        return available_fields

    def get_available_llm_fields(self) -> List[str]:
        """Get fields that have trained LLM models available."""
        available_fields = []

        # Check for LLM models in llm_results directory
        llm_results_dir = os.path.join(os.path.dirname(__file__), "..", "anomaly_detectors", "llm_based", "llm_results")

        for field_name in self.field_mapper.get_available_fields():
            try:
                # Check if LLM model exists for this field
                model_path = os.path.join(llm_results_dir, f"{field_name}_model")
                if os.path.exists(model_path):
                    column_name = self.field_mapper.get_column_name(field_name)
                    if column_name in self.df.columns:
                        available_fields.append(field_name)
                        print(f"   ✅ {field_name}: LLM model available")
                    else:
                        print(f"   ⚠️  {field_name}: LLM model available but column '{column_name}' not in dataset")
                else:
                    print(f"   ❌ {field_name}: No LLM model available")
            except Exception as e:
                print(f"   ❌ {field_name}: Error checking LLM model ({str(e)[:50]}...)")

        return available_fields

    def generate_test_data(self, field_name: str, num_samples: int = 100) -> Tuple[List[str], List[bool]]:
        """
        Generate test data with known anomalies for evaluation.

        Args:
            field_name: Field to generate test data for
            num_samples: Number of samples to generate

        Returns:
            Tuple of (values, is_anomaly_labels)
        """
        column_name = self.field_mapper.get_column_name(field_name)
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset")

        # Get clean values from the dataset
        clean_values = self.df[column_name].dropna().astype(str).unique()
        if len(clean_values) < 10:
            raise ValueError(f"Not enough unique values in {field_name} for testing")

        # Sample clean values
        test_clean = random.sample(clean_values.tolist(), min(num_samples // 2, len(clean_values)))

        # Load injection rules for creating anomalies
        error_rules = []
        anomaly_rules = []

        try:
            error_rules_path = f"validators/error_injection_rules/{field_name}.json"
            error_rules = load_error_rules(error_rules_path)
        except:
            pass

        try:
            anomaly_rules_path = f"anomaly_detectors/anomaly_injection_rules/{field_name}.json"
            anomaly_rules = load_anomaly_rules(anomaly_rules_path)
        except:
            pass

        all_rules = error_rules + anomaly_rules

        # Generate test data
        test_values = []
        test_labels = []

        # Add clean samples
        for clean_value in test_clean:
            test_values.append(clean_value)
            test_labels.append(False)  # Not anomalous

        # Add anomalous samples
        for clean_value in test_clean:
            if all_rules:
                # Create anomalous version
                rule = random.choice(all_rules)
                if 'error_rule' in rule:  # Error injection rule
                    anomalous_value = apply_error_rule(clean_value, rule['error_rule'])
                else:  # Anomaly injection rule
                    anomalous_value = rule.get('corrupted_value', clean_value + "_ANOMALY")

                if anomalous_value != clean_value:
                    test_values.append(anomalous_value)
                    test_labels.append(True)  # Anomalous
            else:
                # Create simple anomalies if no rules available
                anomalous_value = clean_value + "_ANOMALY"
                test_values.append(anomalous_value)
                test_labels.append(True)

        print(f"   📊 Generated {len(test_values)} test samples ({sum(test_labels)} anomalies)")
        return test_values, test_labels

    def evaluate_ml_thresholds(self, field_name: str, thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Evaluate ML detection at different thresholds.

        Args:
            field_name: Field to evaluate
            thresholds: List of thresholds to test

        Returns:
            Dictionary with precision, recall, fpr (false positive rate) for each threshold
        """
        print(f"   🔍 Evaluating {len(thresholds)} ML thresholds for {field_name}...")

        # Load ML model with correct results directory
        results_dir = os.path.join("anomaly_detectors", "results")
        model, column_name, reference_centroid = load_model_for_field(field_name, results_dir=results_dir)

        # Generate test data
        test_values, test_labels = self.generate_test_data(field_name)

        # Get similarity scores for all test values
        results = check_anomalies(model, test_values, threshold=0.0, reference_centroid=reference_centroid)
        similarity_scores = [r['probability_of_correctness'] for r in results]

        # Evaluate each threshold
        precisions = []
        recalls = []
        fprs = []

        for threshold in thresholds:
            # Predict anomalies based on threshold
            predictions = [score < threshold for score in similarity_scores]

            # Calculate metrics
            tp = sum(1 for pred, label in zip(predictions, test_labels) if pred and label)
            fp = sum(1 for pred, label in zip(predictions, test_labels) if pred and not label)
            tn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and not label)
            fn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and label)

            # Calculate precision, recall, and false positive rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            fprs.append(fpr)

        return {
            'thresholds': thresholds,
            'precision': precisions,
            'recall': recalls,
            'fpr': fprs
        }

    def evaluate_llm_thresholds(self, field_name: str, thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Evaluate LLM detection at different thresholds.

        Args:
            field_name: Field to evaluate
            thresholds: List of thresholds to test (log probability thresholds)

        Returns:
            Dictionary with precision, recall, fpr (false positive rate) for each threshold
        """
        print(f"   🔍 Evaluating {len(thresholds)} LLM thresholds for {field_name}...")

        # Create LLM detector
        detector = LLMAnomalyDetector(field_name=field_name, threshold=-10.0)  # Use very low threshold to get all scores

        # Initialize the detector
        detector.learn_patterns(self.df, self.field_mapper.get_column_name(field_name))

        if not detector.is_initialized or not detector.has_trained_model:
            raise ValueError(f"Failed to initialize LLM detector for field {field_name}")

        # Generate test data
        test_values, test_labels = self.generate_test_data(field_name)

        # Get probability scores for all test values
        probability_scores = []
        for value in test_values:
            try:
                # Get raw probability score from the detector
                anomaly = detector._detect_anomaly(value)
                if anomaly:
                    # Extract the raw probability score from details
                    raw_score = anomaly.details.get('probability_score', -10.0)
                else:
                    # If no anomaly detected, use a high probability score
                    raw_score = 0.0
                probability_scores.append(raw_score)
            except Exception as e:
                print(f"      ⚠️  Error processing value '{value[:30]}...': {e}")
                probability_scores.append(0.0)

        # Evaluate each threshold
        precisions = []
        recalls = []
        fprs = []

        for threshold in thresholds:
            # Predict anomalies based on threshold (lower probability = more anomalous)
            predictions = [score < threshold for score in probability_scores]

            # Calculate metrics
            tp = sum(1 for pred, label in zip(predictions, test_labels) if pred and label)
            fp = sum(1 for pred, label in zip(predictions, test_labels) if pred and not label)
            tn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and not label)
            fn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and label)

            # Calculate precision, recall, and false positive rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            fprs.append(fpr)

        return {
            'thresholds': thresholds,
            'precision': precisions,
            'recall': recalls,
            'fpr': fprs
        }

    def plot_precision_recall_curve(self, field_name: str, results: Dict[str, List[float]], detection_type: str = "ML"):
        """Plot precision-recall curve for a field."""
        plt.figure(figsize=(10, 6))

        # Plot precision and recall vs threshold
        plt.plot(results['thresholds'], results['precision'], 'b-', linewidth=2, label='Precision', color='blue')
        plt.plot(results['thresholds'], results['recall'], 'r-', linewidth=2, label='Recall', color='red')

        # Find intersection point
        intersection_idx = None
        for i in range(len(results['thresholds'])):
            if abs(results['precision'][i] - results['recall'][i]) < 0.01:  # Within 1% tolerance
                intersection_idx = i
                break

        if intersection_idx is not None:
            plt.scatter(results['thresholds'][intersection_idx], results['precision'][intersection_idx],
                       color='purple', s=100, zorder=5,
                       label=f"Intersection (t={results['thresholds'][intersection_idx]:.2f})")

        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Precision / Recall', fontsize=12)
        plt.title(f'Precision-Recall vs Threshold: {field_name} ({detection_type})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add performance summary
        max_f1_idx = np.argmax([2 * p * r / (p + r) if (p + r) > 0 else 0
                               for p, r in zip(results['precision'], results['recall'])])
        max_f1_threshold = results['thresholds'][max_f1_idx]
        max_f1_precision = results['precision'][max_f1_idx]
        max_f1_recall = results['recall'][max_f1_idx]

        plt.figtext(0.02, 0.02,
                   f'Best F1: {2 * max_f1_precision * max_f1_recall / (max_f1_precision + max_f1_recall):.3f} '
                   f'(threshold={max_f1_threshold:.2f})',
                   fontsize=10, style='italic')

        # Save plot
        plot_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_precision_recall.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📈 Saved precision-recall curve: {plot_path}")

    def plot_roc_curve(self, field_name: str, results: Dict[str, List[float]], detection_type: str = "ML"):
        """Plot ROC curve for a field."""
        plt.figure(figsize=(10, 6))

        # Plot ROC curve
        plt.plot(results['fpr'], results['recall'], 'r-', linewidth=2, label=f'{detection_type} Detection')

        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

        # Add threshold annotations for key points
        for i, threshold in enumerate(results['thresholds']):
            if i % 5 == 0:  # Annotate every 5th threshold
                plt.annotate(f'{threshold:.2f}',
                           (results['fpr'][i], results['recall'][i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Curve: {field_name} ({detection_type})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Calculate AUC (approximate)
        auc = np.trapz(results['recall'], results['fpr'])
        plt.figtext(0.02, 0.02, f'AUC: {auc:.3f}', fontsize=10, style='italic')

        # Save plot
        plot_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_roc.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📈 Saved ROC curve: {plot_path}")

    def save_metrics(self, field_name: str, results: Dict[str, List[float]], detection_type: str = "ML"):
        """Save detailed metrics to JSON file."""
        # Calculate additional metrics
        f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0
                    for p, r in zip(results['precision'], results['recall'])]

        # Find best threshold for different metrics
        best_f1_idx = np.argmax(f1_scores)
        best_precision_idx = np.argmax(results['precision'])
        best_recall_idx = np.argmax(results['recall'])

        metrics = {
            'field_name': field_name,
            'detection_type': detection_type,
            'thresholds': results['thresholds'],
            'precision': results['precision'],
            'recall': results['recall'],
            'fpr': results['fpr'],
            'f1_scores': f1_scores,
            'best_thresholds': {
                'f1': {
                    'threshold': results['thresholds'][best_f1_idx],
                    'precision': results['precision'][best_f1_idx],
                    'recall': results['recall'][best_f1_idx],
                    'f1': f1_scores[best_f1_idx]
                },
                'precision': {
                    'threshold': results['thresholds'][best_precision_idx],
                    'precision': results['precision'][best_precision_idx],
                    'recall': results['recall'][best_precision_idx],
                    'f1': f1_scores[best_precision_idx]
                },
                'recall': {
                    'threshold': results['thresholds'][best_recall_idx],
                    'precision': results['precision'][best_recall_idx],
                    'recall': results['recall'][best_recall_idx],
                    'f1': f1_scores[best_recall_idx]
                }
            }
        }

        # Save to JSON
        metrics_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"   📊 Saved metrics: {metrics_path}")
        return metrics

    def generate_curves_for_field(self, field_name: str, detection_type: str = "ML", thresholds: List[float] = None):
        """Generate all curves and metrics for a single field."""
        if thresholds is None:
            if detection_type == "ML":
                thresholds = [round(x, 2) for x in np.arange(0.1, 1.0, 0.05)]
            else:  # LLM
                thresholds = [round(x, 2) for x in np.arange(-0.5, 0.1, 0.05)]

        print(f"\n🎯 Generating curves for field: {field_name} ({detection_type})")

        try:
            # Evaluate thresholds based on detection type
            if detection_type == "ML":
                results = self.evaluate_ml_thresholds(field_name, thresholds)
            else:  # LLM
                results = self.evaluate_llm_thresholds(field_name, thresholds)

            # Generate plots
            self.plot_precision_recall_curve(field_name, results, detection_type)
            self.plot_roc_curve(field_name, results, detection_type)

            # Save metrics
            metrics = self.save_metrics(field_name, results, detection_type)

            # Print summary
            best_f1 = metrics['best_thresholds']['f1']
            print(f"   🏆 Best F1 Score: {best_f1['f1']:.3f} (threshold={best_f1['threshold']:.2f})")
            print(f"   📊 Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}")

            return metrics

        except Exception as e:
            print(f"   ❌ Failed to generate curves for {field_name} ({detection_type}): {e}")
            return None

    def generate_all_curves(self, fields: List[str] = None, detection_type: str = "ML", thresholds: List[float] = None):
        """Generate curves for all available fields."""
        if fields is None:
            if detection_type == "ML":
                fields = self.get_available_ml_fields()
            else:  # LLM
                fields = self.get_available_llm_fields()

        if not fields:
            print(f"❌ No fields with {detection_type} models available")
            return

        print(f"🎯 Generating curves for {len(fields)} fields ({detection_type}): {fields}")

        all_metrics = {}

        for field_name in fields:
            metrics = self.generate_curves_for_field(field_name, detection_type, thresholds)
            if metrics:
                all_metrics[field_name] = metrics

        # Generate summary report
        self.generate_summary_report(all_metrics, detection_type)

    def generate_summary_report(self, all_metrics: Dict[str, Any], detection_type: str = "ML"):
        """Generate a summary report comparing all fields."""
        if not all_metrics:
            return

        print(f"\n📊 Summary Report ({detection_type})")
        print("=" * 80)

        # Create summary table
        summary_data = []
        for field_name, metrics in all_metrics.items():
            best_f1 = metrics['best_thresholds']['f1']
            summary_data.append({
                'Field': field_name,
                'Best F1': f"{best_f1['f1']:.3f}",
                'Threshold': f"{best_f1['threshold']:.2f}",
                'Precision': f"{best_f1['precision']:.3f}",
                'Recall': f"{best_f1['recall']:.3f}"
            })

        # Sort by F1 score
        summary_data.sort(key=lambda x: float(x['Best F1']), reverse=True)

        # Print table
        print(f"{'Field':<20} {'Best F1':<10} {'Threshold':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        for row in summary_data:
            print(f"{row['Field']:<20} {row['Best F1']:<10} {row['Threshold']:<10} "
                  f"{row['Precision']:<10} {row['Recall']:<10}")

        # Save summary
        summary_path = os.path.join(self.output_dir, f"{detection_type.lower()}_summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n📁 Summary saved to: {summary_path}")
        print(f"📁 All curves and metrics saved to: {self.output_dir}/")


def entry(data_file=None, detection_type="ml", fields=None, output_dir="detection_curves",
          thresholds=None, brand=None):
    """Entry function for ML curve generation."""

    if not data_file:
        raise ValueError("data_file is required")

    # Handle brand configuration
    if not brand:
        available_brands = get_available_brands()
        if len(available_brands) == 1:
            brand = available_brands[0]
            print(f"Using default brand: {brand}")
        else:
            raise ValueError("Brand must be specified with --brand option")

    brand_config = load_brand_config(brand)
    print(f"Using brand configuration: {brand}")

    # Initialize generator
    generator = DetectionCurveGenerator(data_file, output_dir)
    generator.field_mapper = FieldMapper.from_brand(brand)

    # Generate curves
    generator.generate_all_curves(fields=fields, detection_type=detection_type.upper(), thresholds=thresholds)


def main():
    parser = argparse.ArgumentParser(
        description="Generate precision-recall and ROC curves for ML-based and LLM-based anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
      python ml_curve_generator.py data/your_data.csv --brand your_brand
  python ml_curve_generator.py data/my_data.csv --detection-type llm
  python ml_curve_generator.py data/my_data.csv --fields material color_name category --detection-type ml
  python ml_curve_generator.py data/my_data.csv --output-dir my_curves --thresholds 0.1 0.3 0.5 0.7 0.9
        """
    )

    parser.add_argument("data_file", help="Path to the CSV data file")
    parser.add_argument("--detection-type", choices=["ml", "llm"], default="ml",
                       help="Type of detection to evaluate (default: ml)")
    parser.add_argument("--fields", nargs='+', help="Specific fields to generate curves for (default: all available)")
    parser.add_argument("--output-dir", default="detection_curves", help="Output directory for curves (default: detection_curves)")
    parser.add_argument("--thresholds", nargs='+', type=float,
                       help="Specific thresholds to test (default: ML=0.1-0.95, LLM=-0.5-0.1)")
    parser.add_argument("--brand", help="Brand name (deprecated - uses static config)")

    args = parser.parse_args()

    entry(
        data_file=args.data_file,
        detection_type=args.detection_type,
        fields=args.fields,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        brand=args.brand
    )


if __name__ == "__main__":
    main()
