#!/usr/bin/env python3
"""
Confusion Matrix Analyzer for Data Quality Monitoring

This module provides functionality to calculate and visualize confusion matrices
for the data quality monitoring system, supporting both overall and per-field analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_detector import CellClassification, FieldDetectionResult

from common.field_mapper import FieldMapper


class ConfusionMatrixAnalyzer:
    """Analyzer for calculating and visualizing confusion matrices."""

    def __init__(self, field_mapper: Optional[FieldMapper] = None):
        """
        Initialize the confusion matrix analyzer.

        Args:
            field_mapper: Optional field mapper for column name resolution
        """
        self.field_mapper = field_mapper
        if self.field_mapper is None:
            raise ValueError("field_mapper must be provided")

    def calculate_confusion_matrix(self,
                                 cell_classifications: List[CellClassification],
                                 injection_metadata: Dict[str, List[Dict[str, Any]]],
                                 field_results: Dict[str, FieldDetectionResult],
                                 dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate confusion matrix for overall and per-field performance.

        Args:
            cell_classifications: List of cell classifications from detection
            injection_metadata: Metadata about injected errors/anomalies
            field_results: Detection results for each field
            dataset_size: Number of rows in the dataset (for true negative calculation)

        Returns:
            Dictionary containing confusion matrix data
        """

        # Overall confusion matrix
        overall_cm = self._calculate_overall_confusion_matrix(
            cell_classifications, injection_metadata, field_results, dataset_size
        )

        # Per-field confusion matrices
        per_field_cm = self._calculate_per_field_confusion_matrices(
            cell_classifications, injection_metadata, field_results, dataset_size
        )

        # Per-detection-type confusion matrices
        detection_type_cm = self._calculate_detection_type_confusion_matrices(
            cell_classifications, injection_metadata, field_results, dataset_size
        )

        return {
            "overall": overall_cm,
            "per_field": per_field_cm,
            "per_detection_type": detection_type_cm,
            "summary": self._create_confusion_matrix_summary(overall_cm, per_field_cm, detection_type_cm)
        }

    def _calculate_overall_confusion_matrix(self,
                                          cell_classifications: List[CellClassification],
                                          injection_metadata: Dict[str, List[Dict[str, Any]]],
                                          field_results: Dict[str, FieldDetectionResult],
                                          dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate overall confusion matrix across all fields and detection types."""

        # Create sets of detected and injected issues
        detected_issues = set()
        for classification in cell_classifications:
            key = (classification.row_index, classification.column_name)
            detected_issues.add(key)

        injected_issues = set()
        for field_name, injections in injection_metadata.items():
            for injection in injections:
                row_idx = injection["row_index"]
                column_name = self.field_mapper.get_column_name(field_name)
                key = (row_idx, column_name)
                injected_issues.add(key)

        # Calculate confusion matrix components
        true_positives = detected_issues & injected_issues
        false_positives = detected_issues - injected_issues
        false_negatives = injected_issues - detected_issues

        # For true negatives, we need to consider the total number of cells in the dataset
        # that were analyzed by detection methods. Since we don't have this information
        # directly, we'll estimate based on the fields that were analyzed
        if dataset_size is None:
            raise ValueError("dataset_size is required to compute confusion matrix true negatives")

        total_cells_analyzed = dataset_size * len(field_results)

        # True negatives = total cells analyzed - (TP + FP + FN)
        true_negatives = max(0, total_cells_analyzed - len(true_positives) - len(false_positives) - len(false_negatives))

        # Calculate metrics
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)
        tn = true_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            "matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "accuracy": round(accuracy, 4)
            },
            "counts": {
                "total_detected": len(detected_issues),
                "total_injected": len(injected_issues),
                "total_correctly_detected": tp,
                "total_incorrectly_detected": fp,
                "total_missed": fn
            }
        }

    def _calculate_per_field_confusion_matrices(self,
                                              cell_classifications: List[CellClassification],
                                              injection_metadata: Dict[str, List[Dict[str, Any]]],
                                              field_results: Dict[str, FieldDetectionResult],
                                              dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate confusion matrices for each field."""

        per_field_results = {}

        for field_name in field_results.keys():
            # Get field-specific classifications
            field_classifications = [
                c for c in cell_classifications
                if c.field_name == field_name
            ]

            # Get field-specific injections
            field_injections = injection_metadata.get(field_name, [])

            # Calculate confusion matrix for this field
            field_cm = self._calculate_field_confusion_matrix(
                field_classifications, field_injections, field_name, dataset_size
            )

            per_field_results[field_name] = field_cm

        return per_field_results

    def _calculate_field_confusion_matrix(self,
                                        field_classifications: List[CellClassification],
                                        field_injections: List[Dict[str, Any]],
                                        field_name: str,
                                        dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate confusion matrix for a specific field."""

        # Create sets of detected and injected issues for this field
        detected_issues = set()
        for classification in field_classifications:
            key = (classification.row_index, classification.column_name)
            detected_issues.add(key)

        injected_issues = set()
        for injection in field_injections:
            row_idx = injection["row_index"]
            column_name = self.field_mapper.get_column_name(field_name)
            key = (row_idx, column_name)
            injected_issues.add(key)

        # Calculate confusion matrix components
        true_positives = detected_issues & injected_issues
        false_positives = detected_issues - injected_issues
        false_negatives = injected_issues - detected_issues

        # For true negatives, estimate based on dataset size
        if dataset_size is None:
            dataset_size = 2704  # Default fallback

        total_cells_in_field = dataset_size
        true_negatives = max(0, total_cells_in_field - len(true_positives) - len(false_positives) - len(false_negatives))

        # Calculate metrics
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)
        tn = true_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            "matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "accuracy": round(accuracy, 4)
            },
            "counts": {
                "total_detected": len(detected_issues),
                "total_injected": len(injected_issues),
                "total_correctly_detected": tp,
                "total_incorrectly_detected": fp,
                "total_missed": fn
            }
        }

    def _calculate_detection_type_confusion_matrices(self,
                                                   cell_classifications: List[CellClassification],
                                                   injection_metadata: Dict[str, List[Dict[str, Any]]],
                                                   field_results: Dict[str, FieldDetectionResult],
                                                   dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate confusion matrices for each detection type."""

        detection_types = ["validation", "pattern_based", "ml_based", "llm_based"]
        per_type_results = {}

        for detection_type in detection_types:
            # Get classifications for this detection type
            type_classifications = [
                c for c in cell_classifications
                if c.detection_type == detection_type
            ]

            # Calculate confusion matrix for this detection type
            type_cm = self._calculate_detection_type_confusion_matrix(
                type_classifications, injection_metadata, detection_type, dataset_size
            )

            per_type_results[detection_type] = type_cm

        return per_type_results

    def _calculate_detection_type_confusion_matrix(self,
                                                 type_classifications: List[CellClassification],
                                                 injection_metadata: Dict[str, List[Dict[str, Any]]],
                                                 detection_type: str,
                                                 dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate confusion matrix for a specific detection type."""

        # Create sets of detected issues for this detection type
        detected_issues = set()
        for classification in type_classifications:
            key = (classification.row_index, classification.column_name)
            detected_issues.add(key)

        # Create sets of injected issues that match this detection type
        injected_issues = set()
        for field_name, injections in injection_metadata.items():
            for injection in injections:
                # Map detection type to injection type
                if detection_type == "validation" and injection["injection_type"] == "error":
                    row_idx = injection["row_index"]
                    column_name = self.field_mapper.get_column_name(field_name)
                    key = (row_idx, column_name)
                    injected_issues.add(key)
                elif detection_type in ["pattern_based", "ml_based"] and injection["injection_type"] == "anomaly":
                    row_idx = injection["row_index"]
                    column_name = self.field_mapper.get_column_name(field_name)
                    key = (row_idx, column_name)
                    injected_issues.add(key)

        # Calculate confusion matrix components
        true_positives = detected_issues & injected_issues
        false_positives = detected_issues - injected_issues
        false_negatives = injected_issues - detected_issues

        # For true negatives, estimate based on dataset size
        if dataset_size is None:
            dataset_size = 2704  # Default fallback

        # Assuming multiple fields were analyzed
        total_cells_analyzed = dataset_size * 5  # Assuming 5 fields were analyzed
        true_negatives = max(0, total_cells_analyzed - len(true_positives) - len(false_positives) - len(false_negatives))

        # Calculate metrics
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)
        tn = true_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            "matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "accuracy": round(accuracy, 4)
            },
            "counts": {
                "total_detected": len(detected_issues),
                "total_injected": len(injected_issues),
                "total_correctly_detected": tp,
                "total_incorrectly_detected": fp,
                "total_missed": fn
            }
        }

    def _create_confusion_matrix_summary(self, overall_cm: Dict[str, Any],
                                       per_field_cm: Dict[str, Any],
                                       detection_type_cm: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of all confusion matrix results."""

        # Best performing fields by F1 score
        field_performance = []
        for field_name, field_cm in per_field_cm.items():
            field_performance.append({
                "field_name": field_name,
                "f1_score": field_cm["metrics"]["f1_score"],
                "precision": field_cm["metrics"]["precision"],
                "recall": field_cm["metrics"]["recall"]
            })

        field_performance.sort(key=lambda x: x["f1_score"], reverse=True)

        # Best performing detection types by F1 score
        detection_type_performance = []
        for detection_type, type_cm in detection_type_cm.items():
            detection_type_performance.append({
                "detection_type": detection_type,
                "f1_score": type_cm["metrics"]["f1_score"],
                "precision": type_cm["metrics"]["precision"],
                "recall": type_cm["metrics"]["recall"]
            })

        detection_type_performance.sort(key=lambda x: x["f1_score"], reverse=True)

        return {
            "overall_performance": {
                "f1_score": overall_cm["metrics"]["f1_score"],
                "precision": overall_cm["metrics"]["precision"],
                "recall": overall_cm["metrics"]["recall"],
                "accuracy": overall_cm["metrics"]["accuracy"]
            },
            "best_performing_fields": field_performance[:5],  # Top 5
            "best_performing_detection_types": detection_type_performance,
            "total_fields_analyzed": len(per_field_cm),
            "total_detection_types_analyzed": len(detection_type_cm)
        }

    def create_confusion_matrix_visualizations(self,
                                             confusion_matrix_data: Dict[str, Any],
                                             output_dir: str,
                                             sample_name: str = "confusion_matrix_analysis") -> Dict[str, str]:
        """
        Create and save confusion matrix visualizations.

        Args:
            confusion_matrix_data: Data from calculate_confusion_matrix
            output_dir: Directory to save visualizations
            sample_name: Name for the sample (used in filenames)

        Returns:
            Dictionary with paths to generated visualization files
        """

        os.makedirs(output_dir, exist_ok=True)
        generated_files = {}

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Overall confusion matrix heatmap
        overall_path = self._create_overall_confusion_matrix_heatmap(
            confusion_matrix_data["overall"], output_dir, sample_name
        )
        generated_files["overall_heatmap"] = overall_path

        # 2. Per-field confusion matrix heatmap
        per_field_path = self._create_per_field_confusion_matrix_heatmap(
            confusion_matrix_data["per_field"], output_dir, sample_name
        )
        generated_files["per_field_heatmap"] = per_field_path

        # 3. Detection type confusion matrix heatmap
        detection_type_path = self._create_detection_type_confusion_matrix_heatmap(
            confusion_matrix_data["per_detection_type"], output_dir, sample_name
        )
        generated_files["detection_type_heatmap"] = detection_type_path

        # 4. Performance comparison charts
        performance_path = self._create_performance_comparison_charts(
            confusion_matrix_data, output_dir, sample_name
        )
        generated_files["performance_charts"] = performance_path

        # 5. Summary visualization
        summary_path = self._create_summary_visualization(
            confusion_matrix_data["summary"], output_dir, sample_name
        )
        generated_files["summary_visualization"] = summary_path

        return generated_files

    def _create_overall_confusion_matrix_heatmap(self,
                                               overall_cm: Dict[str, Any],
                                               output_dir: str,
                                               sample_name: str) -> str:
        """Create overall confusion matrix heatmap."""

        # Create confusion matrix array
        matrix_data = np.array([
            [overall_cm["matrix"]["true_positives"], overall_cm["matrix"]["false_positives"]],
            [overall_cm["matrix"]["false_negatives"], overall_cm["matrix"]["true_negatives"]]
        ])

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(matrix_data,
                   annot=True,
                   fmt='d',
                   cmap='Blues',
                   xticklabels=['Predicted Positive', 'Predicted Negative'],
                   yticklabels=['Actual Positive', 'Actual Negative'],
                   ax=ax)

        # Add title and labels
        ax.set_title(f'Overall Confusion Matrix\nF1: {overall_cm["metrics"]["f1_score"]:.3f}, '
                    f'Precision: {overall_cm["metrics"]["precision"]:.3f}, '
                    f'Recall: {overall_cm["metrics"]["recall"]:.3f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)

        # Add metrics text
        metrics_text = (f'Precision: {overall_cm["metrics"]["precision"]:.3f}\n'
                       f'Recall: {overall_cm["metrics"]["recall"]:.3f}\n'
                       f'F1 Score: {overall_cm["metrics"]["f1_score"]:.3f}\n'
                       f'Accuracy: {overall_cm["metrics"]["accuracy"]:.3f}')

        plt.figtext(0.02, 0.02, metrics_text, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{sample_name}_overall_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_per_field_confusion_matrix_heatmap(self,
                                                 per_field_cm: Dict[str, Any],
                                                 output_dir: str,
                                                 sample_name: str) -> str:
        """Create per-field confusion matrix heatmap."""

        if not per_field_cm:
            return ""

        # Prepare data for heatmap
        fields = list(per_field_cm.keys())
        metrics = ['precision', 'recall', 'f1_score']

        # Create data matrix
        data_matrix = np.zeros((len(fields), len(metrics)))
        for i, field in enumerate(fields):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = per_field_cm[field]["metrics"][metric]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(fields) * 0.4)))

        # Create heatmap
        sns.heatmap(data_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   xticklabels=metrics,
                   yticklabels=fields,
                   ax=ax,
                   cbar_kws={'label': 'Score'})

        # Add title and labels
        ax.set_title('Per-Field Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Fields', fontsize=12)

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{sample_name}_per_field_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_detection_type_confusion_matrix_heatmap(self,
                                                      detection_type_cm: Dict[str, Any],
                                                      output_dir: str,
                                                      sample_name: str) -> str:
        """Create detection type confusion matrix heatmap."""

        if not detection_type_cm:
            return ""

        # Prepare data for heatmap
        detection_types = list(detection_type_cm.keys())
        metrics = ['precision', 'recall', 'f1_score']

        # Create data matrix
        data_matrix = np.zeros((len(detection_types), len(metrics)))
        for i, det_type in enumerate(detection_types):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = detection_type_cm[det_type]["metrics"][metric]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(data_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   xticklabels=metrics,
                   yticklabels=[dt.replace('_', ' ').title() for dt in detection_types],
                   ax=ax,
                   cbar_kws={'label': 'Score'})

        # Add title and labels
        ax.set_title('Detection Type Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Detection Types', fontsize=12)

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{sample_name}_detection_type_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_performance_comparison_charts(self,
                                            confusion_matrix_data: Dict[str, Any],
                                            output_dir: str,
                                            sample_name: str) -> str:
        """Create performance comparison charts."""

        # Create subplots for different comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Field performance comparison (F1 scores)
        fields = list(confusion_matrix_data["per_field"].keys())
        f1_scores = [confusion_matrix_data["per_field"][field]["metrics"]["f1_score"] for field in fields]

        bars1 = ax1.bar(range(len(fields)), f1_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Field Performance (F1 Scores)', fontweight='bold')
        ax1.set_xlabel('Fields')
        ax1.set_ylabel('F1 Score')
        ax1.set_xticks(range(len(fields)))
        ax1.set_xticklabels(fields, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. Detection type performance comparison
        detection_types = list(confusion_matrix_data["per_detection_type"].keys())
        det_type_f1 = [confusion_matrix_data["per_detection_type"][dt]["metrics"]["f1_score"] for dt in detection_types]

        bars2 = ax2.bar(range(len(detection_types)), det_type_f1, color='lightcoral', alpha=0.7)
        ax2.set_title('Detection Type Performance (F1 Scores)', fontweight='bold')
        ax2.set_xlabel('Detection Types')
        ax2.set_ylabel('F1 Score')
        ax2.set_xticks(range(len(detection_types)))
        ax2.set_xticklabels([dt.replace('_', ' ').title() for dt in detection_types])
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars2, det_type_f1):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. Precision vs Recall scatter plot for fields
        field_precisions = [confusion_matrix_data["per_field"][field]["metrics"]["precision"] for field in fields]
        field_recalls = [confusion_matrix_data["per_field"][field]["metrics"]["recall"] for field in fields]

        scatter1 = ax3.scatter(field_precisions, field_recalls, s=100, alpha=0.7, c='green')
        ax3.set_title('Field Performance: Precision vs Recall', fontweight='bold')
        ax3.set_xlabel('Precision')
        ax3.set_ylabel('Recall')
        ax3.grid(True, alpha=0.3)

        # Add field labels
        for i, field in enumerate(fields):
            ax3.annotate(field, (field_precisions[i], field_recalls[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 4. Overall metrics comparison
        overall_metrics = confusion_matrix_data["overall"]["metrics"]
        metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        metric_values = [overall_metrics['precision'], overall_metrics['recall'],
                        overall_metrics['f1_score'], overall_metrics['accuracy']]

        bars4 = ax4.bar(metric_names, metric_values, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax4.set_title('Overall Performance Metrics', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars4, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{sample_name}_performance_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_summary_visualization(self,
                                    summary: Dict[str, Any],
                                    output_dir: str,
                                    sample_name: str) -> str:
        """Create summary visualization."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall performance gauge
        overall_f1 = summary["overall_performance"]["f1_score"]

        # Create gauge chart
        gauge_angles = np.linspace(0, np.pi, 100)
        gauge_values = np.linspace(0, 1, 100)

        ax1.plot(np.cos(gauge_angles), np.sin(gauge_angles), 'k-', linewidth=2)
        ax1.fill_between(np.cos(gauge_angles), 0, np.sin(gauge_angles),
                        where=(gauge_values <= overall_f1), alpha=0.6, color='green')
        ax1.fill_between(np.cos(gauge_angles), 0, np.sin(gauge_angles),
                        where=(gauge_values > overall_f1), alpha=0.3, color='lightgray')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(0, 1.2)
        ax1.set_aspect('equal')
        ax1.set_title(f'Overall F1 Score: {overall_f1:.3f}', fontweight='bold', fontsize=14)
        ax1.text(0, 0.3, f'{overall_f1:.3f}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax1.axis('off')

        # 2. Top performing fields
        top_fields = summary["best_performing_fields"][:5]
        if top_fields:
            field_names = [f["field_name"] for f in top_fields]
            field_f1_scores = [f["f1_score"] for f in top_fields]

            bars = ax2.barh(field_names, field_f1_scores, color='lightblue', alpha=0.7)
            ax2.set_title('Top 5 Performing Fields', fontweight='bold')
            ax2.set_xlabel('F1 Score')
            ax2.set_xlim(0, 1)
            ax2.grid(axis='x', alpha=0.3)

            # Add value labels
            for bar, score in zip(bars, field_f1_scores):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=9)

        # 3. Detection type performance
        det_type_perf = summary["best_performing_detection_types"]
        if det_type_perf:
            det_type_names = [d["detection_type"].replace('_', ' ').title() for d in det_type_perf]
            det_type_f1_scores = [d["f1_score"] for d in det_type_perf]

            bars = ax3.bar(det_type_names, det_type_f1_scores, color='lightcoral', alpha=0.7)
            ax3.set_title('Detection Type Performance', fontweight='bold')
            ax3.set_ylabel('F1 Score')
            ax3.set_ylim(0, 1)
            ax3.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, score in zip(bars, det_type_f1_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        Summary Statistics

        Total Fields Analyzed: {summary['total_fields_analyzed']}
        Total Detection Types: {summary['total_detection_types_analyzed']}

        Overall Performance:
        • Precision: {summary['overall_performance']['precision']:.3f}
        • Recall: {summary['overall_performance']['recall']:.3f}
        • F1 Score: {summary['overall_performance']['f1_score']:.3f}
        • Accuracy: {summary['overall_performance']['accuracy']:.3f}

        Best Field: {top_fields[0]['field_name'] if top_fields else 'N/A'}
        (F1: {f"{top_fields[0]['f1_score']:.3f}" if top_fields else "0.000"})

        Best Detection Type: {det_type_perf[0]['detection_type'] if det_type_perf else 'N/A'}
        (F1: {f"{det_type_perf[0]['f1_score']:.3f}" if det_type_perf else "0.000"})
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{sample_name}_summary_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def save_confusion_matrix_report(self,
                                   confusion_matrix_data: Dict[str, Any],
                                   output_dir: str,
                                   sample_name: str = "confusion_matrix_analysis") -> str:
        """
        Save confusion matrix data as JSON report.

        Args:
            confusion_matrix_data: Data from calculate_confusion_matrix
            output_dir: Directory to save the report
            sample_name: Name for the sample

        Returns:
            Path to the saved JSON report
        """

        os.makedirs(output_dir, exist_ok=True)

        # Clean the data for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: clean_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj

        cleaned_data = clean_for_json(confusion_matrix_data)

        # Save to JSON file
        output_path = os.path.join(output_dir, f"{sample_name}_confusion_matrix_report.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        return output_path


def analyze_confusion_matrices(cell_classifications: List[CellClassification],
                             field_results: Dict[str, FieldDetectionResult],
                             injection_metadata: Dict[str, List[Dict[str, Any]]],
                             output_dir: str,
                             sample_name: str = "confusion_matrix_analysis",
                             field_mapper: Optional[FieldMapper] = None,
                             dataset_size: Optional[int] = None) -> Dict[str, str]:
    """
    Convenience function to analyze confusion matrices and generate visualizations.

    Args:
        cell_classifications: List of cell classifications from detection
        field_results: Detection results for each field
        injection_metadata: Metadata about injected errors/anomalies
        output_dir: Directory to save results
        sample_name: Name for the sample
        field_mapper: Optional field mapper
        dataset_size: Number of rows in the dataset (for true negative calculation)

    Returns:
        Dictionary with paths to generated files
    """

    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer(field_mapper)

    # Calculate confusion matrices
    print(f"📊 Calculating confusion matrices...")
    confusion_matrix_data = analyzer.calculate_confusion_matrix(
        cell_classifications, injection_metadata, field_results, dataset_size
    )

    # Generate visualizations
    print(f"📈 Generating confusion matrix visualizations...")
    visualization_files = analyzer.create_confusion_matrix_visualizations(
        confusion_matrix_data, output_dir, sample_name
    )

    # Save JSON report
    print(f"💾 Saving confusion matrix report...")
    json_report_path = analyzer.save_confusion_matrix_report(
        confusion_matrix_data, output_dir, sample_name
    )

    # Combine all generated files
    all_files = {
        "json_report": json_report_path,
        **visualization_files
    }

    return all_files
