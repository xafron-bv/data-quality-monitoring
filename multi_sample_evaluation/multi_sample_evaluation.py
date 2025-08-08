import argparse
import importlib
import json
import os
import sys
import time
import psutil
import tracemalloc
from datetime import datetime

import pandas as pd

# Add the script's directory to the Python path.
# This ensures that top-level modules like 'interfaces.py' are found
# when validator/reporter modules are loaded dynamically from subdirectories.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import Evaluator

import common.debug_config as debug_config

class ExecutionMetrics:
    """Track execution metrics for performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory_mb = 0
        self.process = psutil.Process()
        self.initial_memory = None
        self.tracemalloc_snapshot = None
        self.detection_times = {
            'validation': 0,
            'anomaly': 0,
            'ml': 0,
            'llm': 0,
            'total': 0
        }
        self.sample_metrics = []
        
    def start(self):
        """Start tracking metrics."""
        self.start_time = time.time()
        tracemalloc.start()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
        
    def record_detection_time(self, detection_type, duration):
        """Record time taken for a specific detection type."""
        if detection_type in self.detection_times:
            self.detection_times[detection_type] += duration
            
    def record_sample_metrics(self, sample_idx, metrics):
        """Record metrics for a specific sample."""
        self.sample_metrics.append({
            'sample_idx': sample_idx,
            'metrics': metrics
        })
        
    def stop(self):
        """Stop tracking and calculate final metrics."""
        self.end_time = time.time()
        self.update_peak_memory()
        
        # Get tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'total_execution_time_seconds': self.end_time - self.start_time,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_increase_mb': self.peak_memory_mb - self.initial_memory,
            'tracemalloc_peak_mb': peak / 1024 / 1024,
            'detection_times': self.detection_times,
            'per_sample_avg_time': (self.end_time - self.start_time) / len(self.sample_metrics) if self.sample_metrics else 0,
            'timestamp': datetime.now().isoformat()
        }
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
from anomaly_detectors.llm_based.llm_anomaly_detector import LLMAnomalyDetector

# Import anomaly injection modules
from common.anomaly_injection import AnomalyInjector, load_anomaly_rules
from common.brand_config import get_available_brands, load_brand_config
from common.debug_config import debug_print
from common.error_injection import ErrorInjector, generate_error_samples, load_error_rules
from common.exceptions import ConfigurationError, DataQualityError, FileOperationError, ModelError
from common.field_mapper import FieldMapper


def load_module_class(module_path: str):
    """Dynamically loads a class from a Python file based on a module path string."""
    try:
        # e.g., 'validators.material.validate:MaterialValidator'
        module_name, class_name = module_path.split(':')
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ConfigurationError(
            f"Could not load class from '{module_path}'",
            details={
                'module_path': module_path,
                'original_error': str(e),
                'suggestion': 'Ensure the path is correct and the file contains the class'
            }
        ) from e

def generate_human_readable_performance_summary(evaluation_results):
    """Generate a human-readable performance summary for the evaluation results."""
    if not evaluation_results:
        return {"error": "No evaluation results available"}

    # Calculate overall metrics
    all_tp = []
    all_fp = []
    all_fn = []
    total_anomalies = 0
    total_ml_issues = 0
    total_validation_errors = 0

    for result in evaluation_results:
        # Validation metrics
        true_positives = result.get('true_positive_details', result.get('true_positives', []))
        false_positives = result.get('false_positive_details', result.get('false_positives', []))
        false_negatives = result.get('false_negative_details', result.get('false_negatives', []))

        all_tp.extend(true_positives)
        all_fp.extend(false_positives)
        all_fn.extend(false_negatives)

        # Detection counts
        total_anomalies += len(result.get('anomaly_results', []))
        total_ml_issues += len(result.get('unified_results', []))
        total_validation_errors += len(result.get('validation_results', []))

    # Calculate performance metrics
    tp_count = len(all_tp)
    fp_count = len(all_fp)
    fn_count = len(all_fn)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Determine performance level
    if f1_score >= 0.9:
        performance_level = "Excellent"
    elif f1_score >= 0.7:
        performance_level = "Good"
    elif f1_score >= 0.5:
        performance_level = "Moderate"
    else:
        performance_level = "Needs Improvement"

    # Collect error types detected
    error_types = set()
    for result in evaluation_results:
        for error in result.get('validation_results', []):
            if 'display_message' in error:
                error_type = error['display_message'].split(':')[0]
                error_types.add(error_type)

        for anomaly in result.get('anomaly_results', []):
            if 'display_message' in anomaly:
                error_type = anomaly['display_message'].split(':')[0]
                error_types.add(f"Anomaly: {error_type}")

        for ml_issue in result.get('unified_results', []):
            detection_type = ml_issue.get('detection_type', 'ML Detection')
            error_types.add(f"ML: {detection_type}")

    return {
        "overall_performance": {
            "performance_level": performance_level,
            "f1_score": round(f1_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "interpretation": {
                "f1_score": f"F1 score of {f1_score:.3f} indicates {performance_level.lower()} detection performance",
                "precision": f"Precision of {precision:.3f} means {precision*100:.1f}% of flagged items are actual errors",
                "recall": f"Recall of {recall:.3f} means {recall*100:.1f}% of actual errors were detected"
            }
        },
        "detection_summary": {
            "total_validation_errors": total_validation_errors,
            "total_anomalies_detected": total_anomalies,
            "total_ml_issues_detected": total_ml_issues,
            "total_samples_evaluated": len(evaluation_results),
            "error_types_found": list(error_types)[:10]  # Limit to first 10 for readability
        },
        "detailed_metrics": {
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "accuracy_note": "True positives are correctly identified errors, false positives are incorrectly flagged valid data, false negatives are missed errors"
        }
    }

def generate_cell_coordinate_mapping(evaluation_results, field_name):
    """Generate a mapping of cell coordinates (row, column) for all detected errors and anomalies."""

    def clean_value(value):
        """Clean a value to ensure JSON compatibility."""
        if pd.isna(value):
            return None
        if isinstance(value, float) and (value != value):  # Additional NaN check
            return None
        return value

    cell_mapping = {
        "errors": [],  # Validation errors
        "anomalies": [],  # Pattern-based anomalies
        "ml_issues": []  # ML-detected issues
    }

    for result in evaluation_results:
        # Process validation errors
        for error in result.get('validation_results', []):
            cell_info = {
                "row_index": clean_value(error.get('row_index')),
                "column_name": clean_value(error.get('column_name')),
                "error_data": clean_value(error.get('error_data')),
                "display_message": clean_value(error.get('display_message')),
                "probability": clean_value(error.get('probability', 1.0)),
                "severity": "error",
                "type": "validation"
            }
            cell_mapping["errors"].append(cell_info)

        # Process anomaly results
        for anomaly in result.get('anomaly_results', []):
            cell_info = {
                "row_index": clean_value(anomaly.get('row_index')),
                "column_name": clean_value(anomaly.get('column_name', field_name)),
                "error_data": clean_value(anomaly.get('error_data')),
                "display_message": clean_value(anomaly.get('display_message')),
                "probability": clean_value(anomaly.get('probability', 0.7)),
                "severity": "anomaly",
                "type": "pattern_based"
            }
            cell_mapping["anomalies"].append(cell_info)

        # Process ML results
        for ml_issue in result.get('unified_results', []):
            cell_info = {
                "row_index": clean_value(ml_issue.get('row_index')),
                "column_name": clean_value(ml_issue.get('column_name', field_name)),
                "error_data": clean_value(ml_issue.get('error_data')),
                "display_message": clean_value(ml_issue.get('display_message')),
                "probability": clean_value(ml_issue.get('probability', 0.5)),
                "severity": "anomaly",
                "type": "ml_based"
            }
            cell_mapping["ml_issues"].append(cell_info)

    # Sort by row index for easier navigation
    for category in cell_mapping.values():
        if isinstance(category, list):
            category.sort(key=lambda x: x.get('row_index', 0) if x.get('row_index') is not None else 0)

    # Add summary statistics
    cell_mapping["summary"] = {
        "total_errors": len(cell_mapping["errors"]),
        "total_anomalies": len(cell_mapping["anomalies"]),
        "total_ml_issues": len(cell_mapping["ml_issues"]),
        "total_issues": len(cell_mapping["errors"]) + len(cell_mapping["anomalies"]) + len(cell_mapping["ml_issues"]),
        "affected_rows": len(set(
            [item.get('row_index') for category in cell_mapping.values() if isinstance(category, list) for item in category if item.get('row_index') is not None]
        ))
    }

    return cell_mapping

def generate_summary_report(evaluation_results, output_dir, ignore_fp=False):
    """Generates a summary report of the evaluation results."""
    summary_lines = []
    summary_lines.append("=" * 80)

    # Determine if we have validation results or only anomaly detection
    has_validation = any('validation_performed' in result for result in evaluation_results)
    has_anomalies = any('anomaly_results' in result for result in evaluation_results)
    has_ml_results = any('ml_results' in result and len(result['ml_results']) > 0 for result in evaluation_results)
    has_llm_results = any('llm_results' in result and len(result['llm_results']) > 0 for result in evaluation_results)

    # Build header based on what detection methods were used
    header_parts = []
    if has_validation:
        header_parts.append("VALIDATION")
    if has_anomalies:
        header_parts.append("ANOMALY DETECTION")
    if has_ml_results:
        header_parts.append("ML DETECTION")
    if has_llm_results:
        header_parts.append("LLM DETECTION")
    
    if header_parts:
        summary_lines.append(", ".join(header_parts) + " SUMMARY")
    else:
        summary_lines.append("NO DETECTION RESULTS")

    summary_lines.append("=" * 80)

    all_tp = []
    all_fp = []
    all_fn = []
    total_ignored_errors = 0
    total_ignored_fps = 0
    total_anomalies = 0  # Count total anomalies
    total_ml_issues = 0  # Count total ML detection issues
    total_llm_issues = 0  # Count total LLM detection issues

    for i, result in enumerate(evaluation_results):
        # Handle validation results if present
        if has_validation:
            # Map new field names to old expected names for compatibility
            true_positives = result.get('true_positive_details', result.get('true_positives', []))
            false_positives = result.get('false_positive_details', result.get('false_positives', []))
            false_negatives = result.get('false_negative_details', result.get('false_negatives', []))

            all_tp.extend(true_positives)
            all_fp.extend(false_positives)
            all_fn.extend(false_negatives)

            ignored_errors = result.get('ignored_errors', 0)
            ignored_fps = result.get('ignored_fps', 0)
            total_ignored_errors += ignored_errors
            total_ignored_fps += ignored_fps

            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1_score = result.get('f1_score', 0)

        # Handle anomaly results
        anomalies = result.get('anomaly_results', [])
        total_anomalies += len(anomalies)

        # Handle ML and LLM results
        ml_results = result.get('ml_results', [])
        total_ml_issues += len(ml_results)
        
        llm_results = result.get('llm_results', [])
        total_llm_issues += len(llm_results)

        sample_id = result.get('sample_id', f"Sample {i}")

        # Show the sample only if debug mode is enabled
        if debug_config.is_debug_enabled():
            summary_lines.append(f"\nSample: {sample_id}")

            # Add validation metrics if present
            if has_validation:
                summary_lines.append(f"  - Precision: {precision:.2f}")
                summary_lines.append(f"  - Recall: {recall:.2f}")
                summary_lines.append(f"  - F1 Score: {f1_score:.2f}")
                summary_lines.append(f"  - True Positives: {len(true_positives)}")
                summary_lines.append(f"  - False Positives: {len(false_positives)}")
                summary_lines.append(f"  - False Negatives: {len(false_negatives)}")

                if ignored_errors > 0:
                    summary_lines.append(f"  - Ignored Errors: {ignored_errors}")
                if ignored_fps > 0:
                    summary_lines.append(f"  - Ignored False Positives: {ignored_fps}")

            # Always show anomaly count
            if has_anomalies:
                summary_lines.append(f"  - Anomalies Detected: {len(anomalies)}")

            # Always show ML detection count
            if has_ml_results:
                summary_lines.append(f"  - ML Issues Detected: {len(ml_results)}")

    # Summary section header
    summary_lines.append("\n" + "=" * 60)

    # Add validation metrics section if validation was run
    if has_validation:
        # Calculate overall metrics
        overall_tp = len(all_tp)
        overall_fp = len(all_fp)
        overall_fn = len(all_fn)

        if overall_tp + overall_fp > 0:
            overall_precision = overall_tp / (overall_tp + overall_fp)
        else:
            overall_precision = 0

        if overall_tp + overall_fn > 0:
            overall_recall = overall_tp / (overall_tp + overall_fn)
        else:
            overall_recall = 0

        if overall_precision + overall_recall > 0:
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        else:
            overall_f1 = 0

        summary_lines.append("OVERALL VALIDATION METRICS:")
        summary_lines.append(f"  - Overall Precision: {overall_precision:.2f}")
        summary_lines.append(f"  - Overall Recall: {overall_recall:.2f}")
        summary_lines.append(f"  - Overall F1 Score: {overall_f1:.2f}")
        summary_lines.append(f"  - Total True Positives: {overall_tp}")
        summary_lines.append(f"  - Total False Positives: {overall_fp}")
        summary_lines.append(f"  - Total False Negatives: {overall_fn}")
        summary_lines.append(f"  - Total errors ignored: {total_ignored_errors}")
        summary_lines.append(f"  - Total false positives ignored: {total_ignored_fps}")

    # Add anomaly metrics section if anomaly detection was run
    if has_anomalies:
        if has_validation or has_ml_results:
            summary_lines.append("\nOVERALL ANOMALY DETECTION METRICS:")
        else:
            summary_lines.append("OVERALL ANOMALY DETECTION METRICS:")
        summary_lines.append(f"  - Total Anomalies Detected: {total_anomalies}")

        # List anomaly types and counts if we have anomalies
        if total_anomalies > 0:
            anomaly_types = {}
            for result in evaluation_results:
                for anomaly in result.get('anomaly_results', []):
                    anomaly_type = anomaly.get('display_message', '').split(':')[0]
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1

            summary_lines.append("\nANOMALY TYPES DETECTED:")
            for anomaly_type, count in anomaly_types.items():
                summary_lines.append(f"  - {anomaly_type}: {count}")

    # Add ML metrics section if ML detection was run
    if has_ml_results:
        if has_validation or has_anomalies:
            summary_lines.append("\nOVERALL ML DETECTION METRICS:")
        else:
            summary_lines.append("OVERALL ML DETECTION METRICS:")
        summary_lines.append(f"  - Total ML Issues Detected: {total_ml_issues}")

    # Add LLM metrics section if LLM detection was run
    if has_llm_results:
        if has_validation or has_anomalies or has_ml_results:
            summary_lines.append("\nOVERALL LLM DETECTION METRICS:")
        else:
            summary_lines.append("OVERALL LLM DETECTION METRICS:")
        summary_lines.append(f"  - Total LLM Issues Detected: {total_llm_issues}")

    # Add false negatives section if validation was run
    if has_validation:
        # Only show false negatives section if we actually have validation results
        summary_lines.append(f"\n--- False Negatives (Missed Errors): {len(all_fn)} Total ---")
        if not all_fn:
            summary_lines.append("Excellent! No errors were missed.")
        else:
            fn_by_type = {}
            for fn in all_fn:
                # Handle new structure where fn is a dict with 'injected_error' field
                if isinstance(fn, dict) and 'injected_error' in fn:
                    rule = fn['injected_error'].get('error_rule', 'unknown_rule')
                    injected_data = fn.get('error_data', 'unknown_data')
                elif isinstance(fn, dict):
                    # Fallback for direct structure
                    rule = fn.get('error_rule', 'unknown_rule')
                    injected_data = fn.get('injected_data', fn.get('error_data', 'unknown_data'))
                else:
                    # Handle other formats
                    rule = 'unknown_rule'
                    injected_data = str(fn)

                fn_by_type.setdefault(rule, []).append(injected_data)

            for rule, examples in fn_by_type.items():
                summary_lines.append(f"\n  - Rule: '{rule}' (Missed {len(examples)} times)")
                for ex in list(set(examples))[:3]:
                    summary_lines.append(f"    - Example: '{ex}'")

        # --- False Positives (Incorrectly Flagged) ---
        if not ignore_fp:
            summary_lines.append(f"\n--- False Positives (Incorrectly Flagged): {len(all_fp)} Total ---")
            if not all_fp:
                summary_lines.append("Excellent! No valid data was incorrectly flagged as an error.")
            else:
                fp_by_type = {}
                for fp in all_fp:
                    # Handle dictionaries from new evaluation system
                    if isinstance(fp, dict):
                        display_message = fp.get('display_message', 'Unknown Error')
                        error_data = fp.get('error_data', 'Unknown Data')
                        # Extract error type from display message or use a generic type
                        error_type = display_message.split(':')[0] if ':' in display_message else display_message
                    else:
                        # Handle old-style error objects
                        error_type = getattr(fp, 'error_type', 'Unknown Error')
                        error_data = getattr(fp, 'error_data', 'Unknown Data')

                    fp_by_type.setdefault(error_type, []).append(error_data)

                for error_type, examples in fp_by_type.items():
                    summary_lines.append(f"\n  - Error Type: '{error_type}' (Flagged {len(examples)} times)")
                    for ex in list(set(str(e) for e in examples))[:3]:
                        summary_lines.append(f"    - Example: '{ex}'")

        # Check for potential missing error messages by looking for error formatting issues in reporter output
        error_formatting_issues = set()
        for res in evaluation_results:
            # Look through reported errors for formatting issues
            for fp in res.get('false_positives', []):
                if 'display_message' in fp and fp['display_message'].startswith('Error formatting message:'):
                    if 'error_code' in fp:
                        error_formatting_issues.add(fp['error_code'])

        if error_formatting_issues:
            summary_lines.append("\n[WARNING] The following error codes had formatting issues in their messages:")
            for code in sorted(error_formatting_issues):
                summary_lines.append(f"  - {code}")

    summary_lines.append("\n" + "="*80)

    summary_text = "\n".join(summary_lines)

    print(summary_text)

    # Save reports
    # Remove all code that writes or prints summary, performance_summary, or metadata files
    # summary_path = os.path.join(os.path.dirname(__file__), output_dir, 'summary_report.txt')
    # with open(summary_path, 'w') as f:
    #     f.write(summary_text)

    # Generate human-readable performance summary
    performance_summary = generate_human_readable_performance_summary(evaluation_results)

    # Extract field name from evaluation results for cell coordinate mapping
    field_name = evaluation_results[0].get('field_name', 'unknown') if evaluation_results else 'unknown'
    cell_coordinates = generate_cell_coordinate_mapping(evaluation_results, field_name)

    full_results_path = os.path.join(output_dir, 'full_evaluation_results.json')
    with open(full_results_path, 'w') as f:
        # Convert sets to lists and handle non-JSON serializable objects
        def make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Convert object to dict representation
                return make_serializable(obj.__dict__)
            else:
                # Handle pandas/numpy NaN and null values
                if pd.isna(obj):
                    return None
                # For basic types, pandas/numpy types, etc.
                try:
                    json.dumps(obj)  # Test if it's JSON serializable
                    return obj
                except (TypeError, ValueError):
                    return str(obj)  # Convert to string if not serializable

        # Extract execution metrics if present
        execution_metrics = None
        results_without_metrics = []
        for result in evaluation_results:
            if 'execution_metrics' in result:
                execution_metrics = result['execution_metrics']
            else:
                results_without_metrics.append(result)
        
        serializable_results = make_serializable(results_without_metrics)

        # Create enhanced output with human-readable summary at the top
        enhanced_output = {
            "human_readable_summary": performance_summary,
            "cell_coordinates": cell_coordinates,
            "detailed_evaluation_results": serializable_results
        }
        
        # Add execution metrics if available
        if execution_metrics:
            enhanced_output["execution_metrics"] = execution_metrics

        json.dump(enhanced_output, f, indent=2)

    # Also save just the human-readable summary as a separate file
    # performance_summary_path = os.path.join(os.path.dirname(__file__), output_dir, 'performance_summary.json')
    # with open(performance_summary_path, 'w') as f:
    #     json.dump(performance_summary, f, indent=2)

    # Save cell coordinates mapping as a separate file for the UI
    # cell_coordinates_path = os.path.join(os.path.dirname(__file__), output_dir, 'cell_coordinates.json')
    # with open(cell_coordinates_path, 'w') as f:
    #     json.dump(cell_coordinates, f, indent=2)

    # print(f"\nSummary report saved to '{summary_path}'")
    # print(f"Full JSON results saved to '{full_results_path}'")
    # print(f"Human-readable performance summary saved to '{performance_summary_path}'")
    # print(f"Cell coordinates mapping saved to '{cell_coordinates_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Sample Statistical Evaluation Engine for Data Quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs statistical evaluation by generating multiple samples with errors/anomalies
for a SINGLE field, allowing you to measure the performance of detectors across many variations.

Example Usage:
  python multi_sample_evaluation.py data/source.csv --field material --output-dir results/material_test
  python multi_sample_evaluation.py data/source.csv --field care_instructions --validator care_instructions
  python multi_sample_evaluation.py data/source.csv --field colour_name --anomaly-detector color_name --run anomaly
  python multi_sample_evaluation.py data/source.csv --field material --ml-detector --run all

This will automatically look for:
- Error Injection Rules: validators/error_injection_rules/<validator>.json
- Anomaly Injection Rules: anomaly_detectors/anomaly_injection_rules/<field>.json
- Validator:  validators/<validator>/validate.py (expecting class 'Validator')
- Reporter:   validators/report:Reporter
- Anomaly Detector: anomaly_detectors/<detector>/detect.py (expecting class 'AnomalyDetector')
- Anomaly Reporter: anomaly_detectors/report:AnomalyReporter
- ML Detector: Sentence transformer models for anomaly detection

If --anomaly-detector is not specified, it defaults to the value of --validator.
"""
    )
    parser.add_argument("source_data", help="Path to the source CSV data file.")
    parser.add_argument("--field", required=True, help="The target field to validate in the CSV file (e.g., 'material', 'care_instructions').")
    parser.add_argument("--validator", help="The validator name to use for files and classes (e.g., 'material', 'care_instructions'). Defaults to the field name if not provided.")
    parser.add_argument("--anomaly-detector", help="The anomaly detector name to use (e.g., 'material', 'color_name'). Defaults to the validator name if not provided.")
    parser.add_argument("--ml-detector", action="store_true", help="Enable ML-based anomaly detection using sentence transformers.")
    parser.add_argument("--llm-detector", action="store_true", help="Enable LLM-based anomaly detection.")
    parser.add_argument("--run", choices=["validation", "anomaly", "ml", "llm", "both", "all"], default="both", help="Specify which analysis to run: validation, anomaly detection, ml detection, llm detection, both (validation+anomaly), or all.")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate for evaluation (default: 32).")
    parser.add_argument("--max-errors", type=int, default=3, help="Maximum number of errors to combine in a single sample (default: 3).")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save all evaluation results and generated samples.")
    parser.add_argument("--ignore-errors", nargs='+', default=[], help="A list of error rule names to ignore during evaluation (e.g., inject_unicode_error).")
    parser.add_argument("--ignore-fp", action="store_true", help="If set, false positives will be ignored in the evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for batch processing and detection operations.")

    # Detection configuration options
    parser.add_argument("--validation-threshold", type=float, default=0.0, help="Minimum confidence threshold for validation results (default: 0.0).")
    parser.add_argument("--anomaly-threshold", type=float, default=0.7, help="Minimum confidence threshold for anomaly detection (default: 0.7).")
    parser.add_argument("--ml-threshold", type=float, default=0.7, help="Minimum confidence threshold for ML detection (default: 0.7).")
    parser.add_argument("--llm-threshold", type=float, default=0.6, help="Minimum confidence threshold for LLM detection (default: 0.6).")
    parser.add_argument("--error-probability", type=float, default=0.1, help="Probability of injecting issues (errors or anomalies) in each row (default: 0.1).")
    parser.add_argument("--error-injection-ratio", type=float, default=0.5, help="Ratio of errors vs anomalies when injecting (0.0-1.0, where 1.0 means errors only, 0.0 means anomalies only, default: 0.5).")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing (default: auto-determined based on system).")
    parser.add_argument("--max-workers", type=int, default=7, help="Maximum number of parallel workers (default: 7).")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8, help="Threshold for high confidence detection results (default: 0.8).")

    # Brand configuration options
    parser.add_argument("--brand", help="Brand name (deprecated - uses static config)")
    parser.add_argument("--brand-config", help="Path to brand configuration JSON file (deprecated - uses static config)")

    args = parser.parse_args()

    # Set debug logging based on command-line argument
    if args.debug:
        debug_config.enable_debug()
        print("Debug logging enabled")
    else:
        debug_config.disable_debug()

    # Handle brand configuration
    if not args.brand:
        available_brands = get_available_brands()
        if len(available_brands) == 1:
            args.brand = available_brands[0]
            print(f"Using default brand: {args.brand}")
        elif len(available_brands) > 1:
            raise ConfigurationError(
                f"Multiple brands available. Please specify one with --brand\n"
                f"Available brands: {', '.join(available_brands)}"
            )
        else:
            raise ConfigurationError("No brand configurations found.")

    # Load brand configuration
    try:
        brand_config = load_brand_config(args.brand)
        print(f"Using brand configuration: {args.brand}")
    except FileNotFoundError as e:
        raise ConfigurationError(f"Brand configuration not found: {e}") from e

    # --- Multi-sample evaluation ---
    # --- Derive paths and class names ---
    field_name = args.field

    if args.validator:
        validator_name = args.validator
    else:
        # Convert field name to snake_case if needed (e.g., "Care Instructions" -> "care_instructions")
        validator_name = field_name

    # If anomaly detector name is not provided, default to validator name
    if args.anomaly_detector:
        detector_name = args.anomaly_detector
    else:
        detector_name = validator_name

    # Determine what to run based on the --run argument
    run_validation = args.run in ["validation", "both", "all"]
    run_anomaly = args.run in ["anomaly", "both", "all"]
    run_ml = args.run in ["ml", "all"] or args.ml_detector
    run_llm = args.run in ["llm", "all"] or args.llm_detector

    # If running "both", try to include all methods but keep it simple
    if args.run == "both":
        run_ml = False  # For now, don't auto-include ML with "both" to avoid complexity
        run_llm = False  # Same for LLM

    rules_path = os.path.join('validators', 'error_injection_rules', f'{validator_name}.json')
    validator_module_str = f"validators.{validator_name}.validate:Validator"
    validator_reporter_module_str = f"validators.report:Reporter"
    anomaly_detector_module_str = f"anomaly_detectors.pattern_based.{detector_name}.detect:AnomalyDetector"
    anomaly_reporter_module_str = f"anomaly_detectors.pattern_based.report:AnomalyReporter"

    print(f"--- Evaluation Setup ---")
    print(f"Target field: '{field_name}'")

    if run_validation:
        print(f"Using validator: '{validator_name}'")
        print(f"Rules: {rules_path}")
        print(f"Validator: {validator_module_str}")
        print(f"Validator Reporter: {validator_reporter_module_str}")

    if run_anomaly:
        print(f"Using anomaly detector: '{detector_name}'")
        print(f"Anomaly Detector: {anomaly_detector_module_str}")
        print(f"Anomaly Reporter: {anomaly_reporter_module_str}")

    if run_ml:
        print(f"Using ML-based anomaly detection")
        print(f"ML Detector: MLAnomalyDetector")
    
    if run_llm:
        print(f"Using LLM-based anomaly detection")
        print(f"LLM Detector: LLMAnomalyDetector")

    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load files and modules
    try:
        df = pd.read_csv(args.source_data)
    except FileNotFoundError:
        print(f"Error: Source data file not found at '{args.source_data}'")
        raise FileOperationError(f"Source data file not found at '{args.source_data}'")

    validator = None
    validator_reporter = None
    anomaly_detector = None
    anomaly_reporter = None
    ml_detector = None

    # Load validation components if needed
    if run_validation:
        try:
            rules = load_error_rules(rules_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Could not read or parse rules file at '{rules_path}'.\nDetails: {e}")
            raise FileOperationError(f"Could not read or parse rules file at '{rules_path}'.\nDetails: {e}")

        # Try JSON-based validator first
        try:
            from validators.json_validator import JSONValidator
            from validators.json_reporter import JSONReporter
            
            validator = JSONValidator(validator_name)
            validator_reporter = JSONReporter(validator_name)
        except FileNotFoundError:
            # Fall back to old system
            ValidatorClass = load_module_class(validator_module_str)
            ReporterClass = load_module_class(validator_reporter_module_str)

            validator = ValidatorClass()
            validator_reporter = ReporterClass(validator_name)

    # Load anomaly detection components if needed
    if run_anomaly:
        try:
            AnomalyDetectorClass = load_module_class(anomaly_detector_module_str)
            AnomalyReporterClass = load_module_class(anomaly_reporter_module_str)

            anomaly_detector = AnomalyDetectorClass()
            anomaly_reporter = AnomalyReporterClass(detector_name)
        except Exception as e:
            print(f"Error: Could not load anomaly detector modules.\nDetails: {e}")
            if run_validation or run_ml:
                print("Continuing without pattern-based anomaly detection.")
                run_anomaly = False
            else:
                raise ConfigurationError(f"Could not load anomaly detector modules.\nDetails: {e}")

    # Load ML detection components if needed
    if run_ml:
        try:
            ml_detector = MLAnomalyDetector(
                field_name=validator_name,
                threshold=args.ml_threshold,
            )
            print("ML anomaly detector initialized successfully.")
        except Exception as e:
            print(f"Error: Could not initialize ML anomaly detector.\nDetails: {e}")
            if run_validation or run_anomaly or run_llm:
                print("Continuing without ML-based anomaly detection.")
                run_ml = False
            else:
                raise ConfigurationError(f"Could not initialize ML anomaly detector.\nDetails: {e}")

    # Load LLM detection components if needed
    llm_detector = None
    if run_llm:
        try:
            llm_detector = LLMAnomalyDetector(
                field_name=validator_name,
                threshold=args.llm_threshold,
            )
            print("LLM anomaly detector initialized successfully.")
        except Exception as e:
            print(f"Error: Could not initialize LLM anomaly detector.\nDetails: {e}")
            if run_validation or run_anomaly or run_ml:
                print("Continuing without LLM-based anomaly detection.")
                run_llm = False
            else:
                raise ConfigurationError(f"Could not initialize LLM anomaly detector.\nDetails: {e}")

    # Create field mapper for the brand
    field_mapper = FieldMapper.from_brand(args.brand)

    # Create evaluator with appropriate components
    evaluator = Evaluator(
        high_confidence_threshold=args.high_confidence_threshold,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        validator=validator,
        validator_reporter=validator_reporter,
        anomaly_detector=anomaly_detector,
        anomaly_reporter=anomaly_reporter,
        ml_detector=ml_detector,
        llm_detector=llm_detector,
        field_mapper=field_mapper
    )

    # Initialize metrics tracking
    exec_metrics = ExecutionMetrics()
    exec_metrics.start()
    
    # Run evaluation - Generate comprehensive samples with both errors and anomalies
    print(f"Generating {args.num_samples} samples with errors and anomalies...")
    sample_generation_start = time.time()

    all_samples = []

    # Load anomaly injection rules
    anomaly_rules = []
    try:
        anomaly_rules_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'anomaly_detectors', 'anomaly_injection_rules', f"{validator_name}.json")
        anomaly_rules = load_anomaly_rules(anomaly_rules_path)
        print(f"Loaded {len(anomaly_rules)} anomaly injection rules")
    except Exception:
        print(f"No anomaly injection rules available for {validator_name}")

    # Generate samples with both error injection and anomaly injection
    for i in range(args.num_samples):
        sample_df = df.copy()
        sample_df.name = f"sample_{i}"
        all_injected_items = []

        # Calculate injection probabilities based on ratio
        error_prob = args.error_probability * args.error_injection_ratio
        anomaly_prob = args.error_probability * (1 - args.error_injection_ratio)
        
        # Apply error injection (validation errors) if available
        if run_validation and 'rules' in locals() and error_prob > 0:
            error_injector = ErrorInjector(rules, field_mapper=field_mapper)
            sample_df, error_injections = error_injector.inject_errors(
                sample_df, field_name, max_errors=args.max_errors//2,
                error_probability=error_prob
            )
            all_injected_items.extend(error_injections)

        # Apply anomaly injection (semantic anomalies) if available
        if anomaly_rules and anomaly_prob > 0:
            anomaly_injector = AnomalyInjector(anomaly_rules)
            sample_df, anomaly_injections = anomaly_injector.inject_anomalies(
                sample_df, field_name, max_anomalies=args.max_errors//2,
                anomaly_probability=anomaly_prob
            )
            all_injected_items.extend(anomaly_injections)

        # Save sample
        sample_path = os.path.join(args.output_dir, f"sample_{i}.csv")
        sample_df.to_csv(sample_path, index=False)

        # Save injection metadata
        injections_path = os.path.join(args.output_dir, f"sample_{i}_injected_items.json")
        with open(injections_path, 'w') as f:
            json.dump(all_injected_items, f, indent=2, ensure_ascii=False)

        all_samples.append({
            "data": sample_df,
            "injected_errors": all_injected_items,  # Combined errors and anomalies
            "sample_index": i
        })

    error_samples = all_samples
    sample_generation_time = time.time() - sample_generation_start
    print(f"Generated samples with {sum(len(s['injected_errors']) for s in all_samples)} total injected items in {sample_generation_time:.2f} seconds")
    exec_metrics.update_peak_memory()

    # Run detection methods on each sample
    print(f"Running detection methods on {len(all_samples)} samples...")

    # Use unified approach when running multiple detection methods to avoid duplication
    use_unified = ((run_ml or run_llm) and (run_validation or run_anomaly)) or (run_validation and run_anomaly and (run_ml or run_llm))

    # Process each sample
    evaluation_results = []
    detection_start_time = time.time()
    
    for sample_idx, sample in enumerate(all_samples):
        print(f"\nProcessing sample {sample_idx + 1}/{len(all_samples)}...")
        sample_start_time = time.time()
        
        # Run detection on this sample
        sample_results = evaluator.evaluate_sample(
            sample['data'],
            field_name,
            injected_errors=sample['injected_errors'],
            run_validation=run_validation,
            run_anomaly_detection=run_anomaly,
            use_unified_approach=use_unified,
            validation_threshold=args.validation_threshold,
            anomaly_threshold=args.anomaly_threshold,
            ml_threshold=args.ml_threshold,
            llm_threshold=args.llm_threshold
        )

        # Store sample results
        sample_result = {
            "field_name": field_name,
            "row_count": len(sample['data']),
            "sample_id": sample.get("sample_index", sample_idx),
            "sample_path": f"sample_{sample.get('sample_index', sample_idx)}.csv",
            
            # Store all detection results
            "validation_results": sample_results.get("validation_results", []),
            "anomaly_results": sample_results.get("anomaly_results", []),
            "ml_results": sample_results.get("ml_results", []),
            "llm_results": sample_results.get("llm_results", []),
            "total_validation_errors": sample_results.get("total_validation_errors", 0),
            "total_anomalies": sample_results.get("total_anomalies", 0),
            "total_ml_issues": sample_results.get("total_ml_issues", 0),
            "total_llm_issues": sample_results.get("total_llm_issues", 0),

            # Approach availability flags
            "validation_available": sample_results.get("validation_available", False),
            "anomaly_detection_available": sample_results.get("anomaly_detection_available", False),
            "ml_detection_available": sample_results.get("ml_detection_available", False),
            "llm_detection_available": sample_results.get("llm_detection_available", False),
            "unified_approach_used": sample_results.get("unified_approach_used", False),
        }

        # If we have unified results, copy them too
        if "unified_results" in sample_results:
            sample_result.update({
                "unified_results": sample_results["unified_results"],
                "unified_total_issues": sample_results.get("unified_total_issues", 0),
                "unified_issues_by_type": sample_results.get("unified_issues_by_type", {}),
            })

        # Calculate performance metrics for this sample's injected errors
        if sample.get("injected_errors") and run_validation:
            validation_results_for_metrics = sample_result["validation_results"]
            perf_metrics = evaluator._calculate_metrics(
                sample["data"],  # Use the sample DataFrame with injected errors
                field_name,
                validation_results_for_metrics,
                sample["injected_errors"]
            )
            sample_result.update(perf_metrics)

        # Add flags to indicate what was run
        if not run_validation:
            sample_result["validation_performed"] = False

        # Record sample processing time
        sample_time = time.time() - sample_start_time
        exec_metrics.record_sample_metrics(sample_idx, {
            'processing_time': sample_time,
            'total_detections': (
                sample_result.get("total_validation_errors", 0) +
                sample_result.get("total_anomalies", 0) +
                sample_result.get("total_ml_issues", 0) +
                sample_result.get("total_llm_issues", 0)
            )
        })
        exec_metrics.update_peak_memory()
        
        evaluation_results.append(sample_result)
    
    # Record total detection time
    total_detection_time = time.time() - detection_start_time
    exec_metrics.record_detection_time('total', total_detection_time)
    
    # Print summary of all samples
    print(f"\n{'='*80}")
    print("MULTI-SAMPLE EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(all_samples)} samples")
    
    # Stop metrics tracking and get final metrics
    execution_metrics = exec_metrics.stop()
    
    # Print execution metrics
    print(f"\n{'='*80}")
    print("EXECUTION METRICS")
    print(f"{'='*80}")
    print(f"Total Execution Time: {execution_metrics['total_execution_time_seconds']:.2f} seconds")
    print(f"  - Sample Generation: {sample_generation_time:.2f} seconds")
    print(f"  - Detection Phase: {total_detection_time:.2f} seconds")
    print(f"  - Average per Sample: {execution_metrics['per_sample_avg_time']:.2f} seconds")
    print(f"\nMemory Usage:")
    print(f"  - Peak Memory: {execution_metrics['peak_memory_mb']:.2f} MB")
    print(f"  - Memory Increase: {execution_metrics['memory_increase_mb']:.2f} MB")
    print(f"  - Tracemalloc Peak: {execution_metrics['tracemalloc_peak_mb']:.2f} MB")
    
    # Add metrics to results
    evaluation_results.append({
        'execution_metrics': execution_metrics
    })
    
    # Report results
    generate_summary_report(evaluation_results, args.output_dir, args.ignore_fp)


if __name__ == '__main__':
    main()
