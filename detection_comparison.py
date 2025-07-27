#!/usr/bin/env python3
"""
Detection Comparison Script

This script compares ML-based and LLM-based anomaly detection methods
side by side for the same fields, generating comparison plots and metrics.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_curve_generator import DetectionCurveGenerator


class DetectionComparison:
    """Compares ML and LLM detection methods side by side."""
    
    def __init__(self, data_file: str, output_dir: str = "detection_comparison"):
        """
        Initialize the comparison.
        
        Args:
            data_file: Path to the CSV data file
            output_dir: Directory to save the comparison results
        """
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_common_fields(self) -> List[str]:
        """Get fields that have both ML and LLM models available."""
        generator = DetectionCurveGenerator(self.data_file, self.output_dir)
        
        ml_fields = set(generator.get_available_ml_fields())
        llm_fields = set(generator.get_available_llm_fields())
        
        common_fields = ml_fields.intersection(llm_fields)
        
        print(f"📊 Available fields:")
        print(f"   ML fields: {sorted(ml_fields)}")
        print(f"   LLM fields: {sorted(llm_fields)}")
        print(f"   Common fields: {sorted(common_fields)}")
        
        return sorted(common_fields)
    
    def compare_field(self, field_name: str) -> Dict[str, Any]:
        """Compare ML and LLM detection for a single field."""
        print(f"\n🎯 Comparing detection methods for field: {field_name}")
        
        generator = DetectionCurveGenerator(self.data_file, self.output_dir)
        
        # Generate curves for both methods
        ml_metrics = generator.generate_curves_for_field(field_name, "ML")
        llm_metrics = generator.generate_curves_for_field(field_name, "LLM")
        
        if not ml_metrics or not llm_metrics:
            print(f"   ❌ Failed to generate metrics for {field_name}")
            return None
        
        # Create comparison plots
        self.plot_comparison_curves(field_name, ml_metrics, llm_metrics)
        
        # Create comparison summary
        comparison = {
            'field_name': field_name,
            'ml_metrics': ml_metrics,
            'llm_metrics': llm_metrics,
            'comparison': self.compare_metrics(ml_metrics, llm_metrics)
        }
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, f"{field_name}_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"   📊 Saved comparison: {comparison_path}")
        return comparison
    
    def plot_comparison_curves(self, field_name: str, ml_metrics: Dict, llm_metrics: Dict):
        """Create comparison plots for ML vs LLM."""
        
        # Precision-Recall comparison
        plt.figure(figsize=(12, 5))
        
        # ML curves
        plt.subplot(1, 2, 1)
        plt.plot(ml_metrics['thresholds'], ml_metrics['precision'], 'b-', linewidth=2, label='ML Precision', color='blue')
        plt.plot(ml_metrics['thresholds'], ml_metrics['recall'], 'b--', linewidth=2, label='ML Recall', color='blue', alpha=0.7)
        
        # LLM curves
        plt.plot(llm_metrics['thresholds'], llm_metrics['precision'], 'r-', linewidth=2, label='LLM Precision', color='red')
        plt.plot(llm_metrics['thresholds'], llm_metrics['recall'], 'r--', linewidth=2, label='LLM Recall', color='red', alpha=0.7)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Precision / Recall', fontsize=12)
        plt.title(f'Precision-Recall Comparison: {field_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # ROC comparison
        plt.subplot(1, 2, 2)
        plt.plot(ml_metrics['fpr'], ml_metrics['recall'], 'b-', linewidth=2, label='ML Detection', color='blue')
        plt.plot(llm_metrics['fpr'], llm_metrics['recall'], 'r-', linewidth=2, label='LLM Detection', color='red')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Comparison: {field_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Calculate AUCs
        ml_auc = np.trapz(ml_metrics['recall'], ml_metrics['fpr'])
        llm_auc = np.trapz(llm_metrics['recall'], llm_metrics['fpr'])
        
        plt.figtext(0.02, 0.02, f'ML AUC: {ml_auc:.3f}, LLM AUC: {llm_auc:.3f}', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"{field_name}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved comparison plot: {plot_path}")
    
    def compare_metrics(self, ml_metrics: Dict, llm_metrics: Dict) -> Dict[str, Any]:
        """Compare the best metrics between ML and LLM."""
        
        ml_best_f1 = ml_metrics['best_thresholds']['f1']
        llm_best_f1 = llm_metrics['best_thresholds']['f1']
        
        comparison = {
            'best_f1_comparison': {
                'ml': {
                    'f1': ml_best_f1['f1'],
                    'precision': ml_best_f1['precision'],
                    'recall': ml_best_f1['recall'],
                    'threshold': ml_best_f1['threshold']
                },
                'llm': {
                    'f1': llm_best_f1['f1'],
                    'precision': llm_best_f1['precision'],
                    'recall': llm_best_f1['recall'],
                    'threshold': llm_best_f1['threshold']
                },
                'winner': 'ML' if ml_best_f1['f1'] > llm_best_f1['f1'] else 'LLM',
                'f1_difference': abs(ml_best_f1['f1'] - llm_best_f1['f1'])
            },
            'auc_comparison': {
                'ml_auc': np.trapz(ml_metrics['recall'], ml_metrics['fpr']),
                'llm_auc': np.trapz(llm_metrics['recall'], llm_metrics['fpr']),
                'winner': 'ML' if np.trapz(ml_metrics['recall'], ml_metrics['fpr']) > np.trapz(llm_metrics['recall'], llm_metrics['fpr']) else 'LLM'
            }
        }
        
        return comparison
    
    def generate_comparison_report(self, comparisons: Dict[str, Any]):
        """Generate a comprehensive comparison report."""
        if not comparisons:
            return
        
        print(f"\n📊 Detection Method Comparison Report")
        print("=" * 100)
        
        # Create comparison table
        comparison_data = []
        for field_name, comparison in comparisons.items():
            if comparison:
                f1_comp = comparison['comparison']['best_f1_comparison']
                auc_comp = comparison['comparison']['auc_comparison']
                
                comparison_data.append({
                    'Field': field_name,
                    'ML F1': f"{f1_comp['ml']['f1']:.3f}",
                    'LLM F1': f"{f1_comp['llm']['f1']:.3f}",
                    'F1 Winner': f1_comp['winner'],
                    'ML AUC': f"{auc_comp['ml_auc']:.3f}",
                    'LLM AUC': f"{auc_comp['llm_auc']:.3f}",
                    'AUC Winner': auc_comp['winner']
                })
        
        # Sort by F1 difference
        comparison_data.sort(key=lambda x: abs(float(x['ML F1']) - float(x['LLM F1'])), reverse=True)
        
        # Print table
        print(f"{'Field':<20} {'ML F1':<8} {'LLM F1':<8} {'F1 Winner':<10} {'ML AUC':<8} {'LLM AUC':<8} {'AUC Winner':<10}")
        print("-" * 100)
        for row in comparison_data:
            print(f"{row['Field']:<20} {row['ML F1']:<8} {row['LLM F1']:<8} {row['F1 Winner']:<10} "
                  f"{row['ML AUC']:<8} {row['LLM AUC']:<8} {row['AUC Winner']:<10}")
        
        # Calculate overall statistics
        ml_wins_f1 = sum(1 for row in comparison_data if row['F1 Winner'] == 'ML')
        llm_wins_f1 = sum(1 for row in comparison_data if row['F1 Winner'] == 'LLM')
        ml_wins_auc = sum(1 for row in comparison_data if row['AUC Winner'] == 'ML')
        llm_wins_auc = sum(1 for row in comparison_data if row['AUC Winner'] == 'LLM')
        
        print(f"\n📈 Overall Statistics:")
        print(f"   F1 Score Wins: ML={ml_wins_f1}, LLM={llm_wins_f1}")
        print(f"   AUC Wins: ML={ml_wins_auc}, LLM={llm_wins_auc}")
        
        # Save comprehensive report
        report_path = os.path.join(self.output_dir, "comprehensive_comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        print(f"\n📁 Comprehensive report saved to: {report_path}")
        print(f"📁 All comparison files saved to: {self.output_dir}/")
    
    def run_comparison(self, fields: List[str] = None):
        """Run comparison for specified fields or all common fields."""
        if fields is None:
            fields = self.get_common_fields()
        
        if not fields:
            print("❌ No fields with both ML and LLM models available")
            return
        
        print(f"🎯 Running comparison for {len(fields)} fields: {fields}")
        
        comparisons = {}
        
        for field_name in fields:
            comparison = self.compare_field(field_name)
            if comparison:
                comparisons[field_name] = comparison
        
        # Generate comprehensive report
        self.generate_comparison_report(comparisons)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML-based and LLM-based anomaly detection methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
      python detection_comparison.py data/your_data.csv
  python detection_comparison.py data/my_data.csv --fields material color_name
        """
    )
    
    parser.add_argument("data_file", help="Path to the CSV data file")
    parser.add_argument("--fields", nargs='+', help="Specific fields to compare (default: all common fields)")
    parser.add_argument("--output-dir", default="detection_comparison", help="Output directory for comparison (default: detection_comparison)")
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = DetectionComparison(args.data_file, args.output_dir)
    
    # Run comparison
    comparison.run_comparison(fields=args.fields)


if __name__ == "__main__":
    main() 