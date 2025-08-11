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
        self.df = None
        self.field_mapper = None
        self.brand_config = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def load_data(self, brand_name: str = None):
        """Load the dataset and brand configuration."""
        # Load dataset
        self.df = pd.read_csv(self.data_file)
        print(f"📊 Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Load brand configuration
        if brand_name:
            self.brand_config = load_brand_config(brand_name)
        else:
            # Use a default or static configuration
            self.brand_config = {
                "ml_fields": ["material", "color_name", "product_name", "category"],
                "llm_fields": ["material", "care_instructions", "description_short_1"]
            }
        
        # Initialize field mapper
        self.field_mapper = FieldMapper.from_brand(brand_name) if brand_name else FieldMapper({})
        
    def get_available_fields(self, detection_type: str = "ML") -> List[str]:
        """Get list of available fields for the detection type."""
        # Define default fields for each detection type
        default_ml_fields = ["material", "color_name", "product_name", "category", "size"]
        default_llm_fields = ["material", "care_instructions", "description_short_1"]
        
        if isinstance(self.brand_config, dict):
            # Using static config
            if detection_type == "ML":
                fields = self.brand_config.get("ml_fields", default_ml_fields)
            else:  # LLM
                fields = self.brand_config.get("llm_fields", default_llm_fields)
        else:
            # Using BrandConfig object - use defaults or enabled_fields
            if hasattr(self.brand_config, 'enabled_fields') and self.brand_config.enabled_fields:
                # Filter by detection type based on field characteristics
                all_fields = self.brand_config.enabled_fields
                if detection_type == "ML":
                    # ML works better with categorical/structured fields
                    fields = [f for f in all_fields if f in default_ml_fields]
                else:  # LLM
                    # LLM works better with text fields
                    fields = [f for f in all_fields if f in default_llm_fields]
            else:
                # Use defaults
                fields = default_ml_fields if detection_type == "ML" else default_llm_fields
            
        # Filter fields that exist in the dataset
        available_fields = []
        for field in fields:
            column_name = self.field_mapper.get_column_name(field)
            if column_name in self.df.columns:
                available_fields.append(field)
                
        return available_fields
    
    def generate_test_data(self, field_name: str, num_samples: int = 100) -> Tuple[List[str], List[bool]]:
        """
        Generate test data with injected anomalies.
        
        Args:
            field_name: Field to generate test data for
            num_samples: Number of test samples to generate
            
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
            error_rules_path = f"validators/error_injection_rules/baseline/{field_name}.json"
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

        # Add anomalous samples with more realistic anomalies
        anomaly_patterns = self._get_anomaly_patterns(field_name)
        
        for i, clean_value in enumerate(test_clean):
            if all_rules and i < len(all_rules):
                # Use existing rules
                rule = all_rules[i % len(all_rules)]
                if 'error_rule' in rule:  # Error injection rule
                    anomalous_value = apply_error_rule(clean_value, rule['error_rule'])
                else:  # Anomaly injection rule
                    anomalous_value = rule.get('corrupted_value', clean_value + "_ANOMALY")
            else:
                # Create realistic anomalies based on field type
                anomaly_pattern = anomaly_patterns[i % len(anomaly_patterns)]
                anomalous_value = anomaly_pattern(clean_value)

            if anomalous_value != clean_value:
                test_values.append(anomalous_value)
                test_labels.append(True)  # Anomalous

        print(f"   📊 Generated {len(test_values)} test samples ({sum(test_labels)} anomalies)")
        return test_values, test_labels
    
    def _get_anomaly_patterns(self, field_name: str) -> List[callable]:
        """Get anomaly generation patterns based on field type."""
        if field_name == "material":
            return [
                lambda x: "100% Premium Beef",
                lambda x: "85% LED - 15% Resistor", 
                lambda x: "100% Bacterial Culture",
                lambda x: "90% Solar Panel - 10% Glass",
                lambda x: "This is not a material at all",
                lambda x: x.replace("%", "#"),
                lambda x: x + " (INVALID)",
                lambda x: "Random gibberish text",
                lambda x: "!@#$%^&*()",
                lambda x: "1234567890"
            ]
        elif field_name == "color_name":
            return [
                lambda x: "Bright Darkness",
                lambda x: "Invisible Purple",
                lambda x: x + "123",
                lambda x: "RGB(999,999,999)",
                lambda x: "Not a color",
                lambda x: "#GGGGGG",
                lambda x: "Color of the wind"
            ]
        else:
            # Generic anomaly patterns
            return [
                lambda x: x + "_ANOMALY",
                lambda x: "INVALID_" + x,
                lambda x: x.replace(" ", "_"),
                lambda x: x[::-1],  # Reverse
                lambda x: "123" + x,
                lambda x: x + "###",
                lambda x: ""  # Empty
            ]

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
        model, column_name, reference_centroid = load_model_for_field(field_name, models_dir=results_dir)

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

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

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
            thresholds: List of thresholds to test (positive anomaly score thresholds)

        Returns:
            Dictionary with precision, recall, fpr (false positive rate) for each threshold
        """
        print(f"   🔍 Evaluating {len(thresholds)} LLM thresholds for {field_name}...")

        # Create LLM detector with very low threshold to detect everything
        detector = LLMAnomalyDetector(field_name=field_name, threshold=0.0)

        # Generate test data
        test_values, test_labels = self.generate_test_data(field_name)
        
        # Create DataFrame for bulk detection
        test_df = pd.DataFrame({field_name: test_values})
        column_name = self.field_mapper.get_column_name(field_name)
        
        # Get detections with raw scores
        print(f"      🤖 Running LLM detection on {len(test_values)} samples...")
        detections = detector.bulk_detect(test_df, column_name, batch_size=50, max_workers=1)
        
        # Extract probability scores - we need to extract the raw anomaly scores
        probability_scores = []
        for i, detection in enumerate(detections):
            if detection is not None:
                # Detection found - extract the raw anomaly score from details
                if 'anomaly_score' in detection.details:
                    score = detection.details['anomaly_score']
                else:
                    # Fall back to normalized probability * 10
                    score = detection.probability * 10.0
                probability_scores.append(score)
            else:
                # No detection - assign zero score (normal)
                probability_scores.append(0.0)
        
        print(f"      📊 Score range: [{min(probability_scores):.4f}, {max(probability_scores):.4f}]")
        print(f"      📊 Mean score: {np.mean(probability_scores):.4f}, Std: {np.std(probability_scores):.4f}")

        # Evaluate each threshold
        precisions = []
        recalls = []
        fprs = []

        for threshold in thresholds:
            # Predict anomalies based on threshold (higher score = more anomalous)
            predictions = [score > threshold for score in probability_scores]

            # Calculate metrics
            tp = sum(1 for pred, label in zip(predictions, test_labels) if pred and label)
            fp = sum(1 for pred, label in zip(predictions, test_labels) if pred and not label)
            tn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and not label)
            fn = sum(1 for pred, label in zip(predictions, test_labels) if not pred and label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            fprs.append(fpr)
            
            # Debug output for key thresholds
            if threshold in [0.0001, 0.001, 0.01, 0.1]:
                print(f"      Threshold {threshold:.4f}: TP={tp}, FP={fp}, TN={tn}, FN={fn} | Prec={precision:.3f}, Rec={recall:.3f}")

        return {
            'thresholds': thresholds,
            'precision': precisions,
            'recall': recalls,
            'fpr': fprs,
            'probability_scores': probability_scores,
            'test_labels': test_labels
        }

    def plot_precision_recall_curve(self, field_name: str, results: Dict[str, List[float]], detection_type: str = "ML"):
        """Plot precision-recall curve."""
        plt.figure(figsize=(10, 8))
        
        # Plot precision and recall vs threshold
        plt.subplot(2, 1, 1)
        plt.plot(results['thresholds'], results['precision'], 'b-', linewidth=2, label='Precision', color='blue')
        plt.plot(results['thresholds'], results['recall'], 'r-', linewidth=2, label='Recall', color='red')
        
        # Calculate F1 scores
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                     for p, r in zip(results['precision'], results['recall'])]
        plt.plot(results['thresholds'], f1_scores, 'g--', linewidth=2, label='F1 Score', color='green')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{detection_type} Detection Performance vs Threshold - {field_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot precision-recall curve
        plt.subplot(2, 1, 2)
        plt.plot(results['recall'], results['precision'], 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {field_name}')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        plt.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5)
        
        # Calculate and display AUC
        auc = np.trapz(results['precision'], results['recall'])
        plt.text(0.6, 0.2, f'AUC: {auc:.3f}', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_precision_recall.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved precision-recall curve: {output_path}")

    def plot_roc_curve(self, field_name: str, results: Dict[str, List[float]], detection_type: str = "ML"):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 8))
        
        # Plot ROC curve
        plt.plot(results['fpr'], results['recall'], 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        
        # Calculate and display AUC
        auc = np.trapz(results['recall'], results['fpr'])
        plt.text(0.6, 0.2, f'AUC: {auc:.3f}', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC Curve - {field_name} ({detection_type})')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_roc.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved ROC curve: {output_path}")
    
    def plot_score_distribution(self, field_name: str, scores: List[float], labels: List[bool], detection_type: str = "LLM"):
        """Plot distribution of scores for normal vs anomalous samples."""
        plt.figure(figsize=(10, 6))
        
        # Separate scores by label
        normal_scores = [s for s, l in zip(scores, labels) if not l]
        anomaly_scores = [s for s, l in zip(scores, labels) if l]
        
        # Plot histograms
        plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=30, alpha=0.5, label='Anomaly', color='red', density=True)
        
        plt.xlabel(f'{detection_type} Score')
        plt.ylabel('Density')
        plt.title(f'Score Distribution - {field_name} ({detection_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for mean values
        plt.axvline(np.mean(normal_scores), color='blue', linestyle='--', linewidth=2, alpha=0.7)
        plt.axvline(np.mean(anomaly_scores), color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add text with statistics
        stats_text = f"Normal: μ={np.mean(normal_scores):.4f}, σ={np.std(normal_scores):.4f}\n"
        stats_text += f"Anomaly: μ={np.mean(anomaly_scores):.4f}, σ={np.std(anomaly_scores):.4f}"
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        output_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_score_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved score distribution: {output_path}")
    
    def save_metrics(self, field_name: str, results: Dict[str, Any], detection_type: str = "ML") -> Dict[str, Any]:
        """Save evaluation metrics to JSON file."""
        # Calculate F1 scores
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                     for p, r in zip(results['precision'], results['recall'])]
        
        # Find best thresholds
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
        
        # Add score statistics if available
        if 'probability_scores' in results:
            normal_scores = [s for s, l in zip(results['probability_scores'], results['test_labels']) if not l]
            anomaly_scores = [s for s, l in zip(results['probability_scores'], results['test_labels']) if l]
            
            metrics['score_statistics'] = {
                'normal': {
                    'mean': np.mean(normal_scores) if normal_scores else 0,
                    'std': np.std(normal_scores) if normal_scores else 0,
                    'min': min(normal_scores) if normal_scores else 0,
                    'max': max(normal_scores) if normal_scores else 0
                },
                'anomaly': {
                    'mean': np.mean(anomaly_scores) if anomaly_scores else 0,
                    'std': np.std(anomaly_scores) if anomaly_scores else 0,
                    'min': min(anomaly_scores) if anomaly_scores else 0,
                    'max': max(anomaly_scores) if anomaly_scores else 0
                },
                'separation': (np.mean(anomaly_scores) - np.mean(normal_scores)) if (normal_scores and anomaly_scores) else 0
            }
        
        output_path = os.path.join(self.output_dir, f"{field_name}_{detection_type.lower()}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   📊 Saved metrics: {output_path}")
        print(f"   🏆 Best F1 Score: {f1_scores[best_f1_idx]:.3f} (threshold={results['thresholds'][best_f1_idx]:.2f})")
        print(f"   📊 Precision: {results['precision'][best_f1_idx]:.3f}, Recall: {results['recall'][best_f1_idx]:.3f}")
        
        return metrics

    def generate_curves_for_field(self, field_name: str, detection_type: str = "ML", thresholds: List[float] = None):
        """Generate all curves and metrics for a single field."""
        if thresholds is None:
            if detection_type == "ML":
                thresholds = [round(x, 2) for x in np.arange(0.1, 1.0, 0.05)]
            else:  # LLM - adjusted for positive anomaly scores
                thresholds = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

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
            
            # Plot score distribution for LLM
            if detection_type == "LLM" and 'probability_scores' in results:
                self.plot_score_distribution(field_name, results['probability_scores'], results['test_labels'], detection_type)

            # Save metrics
            metrics = self.save_metrics(field_name, results, detection_type)

            return metrics
        except Exception as e:
            print(f"   ❌ Error generating curves for {field_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_summary_report(self, all_metrics: List[Dict[str, Any]], detection_type: str):
        """Generate a summary report across all fields."""
        summary = {
            'detection_type': detection_type,
            'fields': [],
            'best_overall_threshold': None
        }
        
        # Collect best F1 scores for each field
        for metrics in all_metrics:
            if metrics:
                field_summary = {
                    'field_name': metrics['field_name'],
                    'best_f1': metrics['best_thresholds']['f1'],
                    'best_precision': metrics['best_thresholds']['precision'],
                    'best_recall': metrics['best_thresholds']['recall']
                }
                
                # Add score statistics if available
                if 'score_statistics' in metrics:
                    field_summary['score_separation'] = metrics['score_statistics']['separation']
                
                summary['fields'].append(field_summary)
        
        # Find overall best threshold (average of best thresholds)
        if summary['fields']:
            avg_threshold = np.mean([f['best_f1']['threshold'] for f in summary['fields']])
            summary['best_overall_threshold'] = round(avg_threshold, 4)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"{detection_type.lower()}_summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary table
        print(f"\n📊 Summary Report ({detection_type})")
        print("=" * 80)
        print(f"{'Field':<20} {'Best F1':<10} {'Threshold':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        
        for field in summary['fields']:
            print(f"{field['field_name']:<20} "
                  f"{field['best_f1']['f1']:<10.3f} "
                  f"{field['best_f1']['threshold']:<10.2f} "
                  f"{field['best_f1']['precision']:<10.3f} "
                  f"{field['best_f1']['recall']:<10.3f}")
        
        if summary['best_overall_threshold']:
            print(f"\n📌 Recommended overall threshold: {summary['best_overall_threshold']}")
        
        print(f"\n📁 Summary saved to: {summary_path}")

def entry(data_file: str, detection_type: str = "ml", fields: List[str] = None,
          output_dir: str = "detection_curves", thresholds: List[float] = None, brand: str = None):
    """
    Main entry point for the curve generator.
    
    Args:
        data_file: Path to CSV data file
        detection_type: Type of detection ('ml' or 'llm')
        fields: Specific fields to evaluate (default: all available)
        output_dir: Output directory for curves
        thresholds: Custom thresholds to test
        brand: Brand name (optional)
    """
    # Initialize generator
    generator = DetectionCurveGenerator(data_file, output_dir)
    
    # Load data
    if brand:
        print(f"Using brand configuration: {brand}")
    generator.load_data(brand)
    
    # Determine fields to evaluate
    detection_type = detection_type.upper()
    available_fields = generator.get_available_fields(detection_type)
    
    if fields:
        # Filter to requested fields that are available
        fields_to_evaluate = [f for f in fields if f in available_fields]
        if not fields_to_evaluate:
            print(f"❌ None of the requested fields {fields} are available for {detection_type} detection")
            return
    else:
        fields_to_evaluate = available_fields
    
    print(f"🎯 Generating curves for {len(fields_to_evaluate)} fields ({detection_type}): {fields_to_evaluate}")
    
    # Generate curves for each field
    all_metrics = []
    for field in fields_to_evaluate:
        metrics = generator.generate_curves_for_field(field, detection_type, thresholds)
        all_metrics.append(metrics)
    
    # Generate summary report
    generator.generate_summary_report(all_metrics, detection_type)
    
    print(f"\n📁 All curves and metrics saved to: {output_dir}/")


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
                       help="Specific thresholds to test (default: ML=0.1-0.95, LLM=0.00001-1.0)")
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
