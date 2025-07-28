import pandas as pd
import os
import json
import sys
import argparse
import importlib

# Add the script's directory to the Python path.
# This ensures that top-level modules like 'interfaces.py' are found
# when validator/reporter modules are loaded dynamically from subdirectories.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluator import Evaluator
from error_injection import generate_error_samples, load_error_rules, ErrorInjector
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
import debug_config
from common.exceptions import DataQualityError, ConfigurationError, FileOperationError, ModelError
from common.field_mapper import FieldMapper

# Import anomaly injection modules
from anomaly_detectors.anomaly_injection import load_anomaly_rules, AnomalyInjector
from brand_config import load_brand_config, get_available_brands


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
    has_ml_results = any('unified_results' in result for result in evaluation_results)
    
    if has_validation and has_anomalies and has_ml_results:
        summary_lines.append("VALIDATION, ANOMALY DETECTION, AND ML DETECTION SUMMARY")
    elif has_validation and has_anomalies:
        summary_lines.append("VALIDATION AND ANOMALY DETECTION SUMMARY")
    elif has_validation and has_ml_results:
        summary_lines.append("VALIDATION AND ML DETECTION SUMMARY")
    elif has_anomalies and has_ml_results:
        summary_lines.append("ANOMALY AND ML DETECTION SUMMARY")
    elif has_validation:
        summary_lines.append("VALIDATION EVALUATION SUMMARY")
    elif has_ml_results:
        summary_lines.append("ML DETECTION SUMMARY")
    else:
        summary_lines.append("ANOMALY DETECTION SUMMARY")
    
    summary_lines.append("=" * 80)
    
    all_tp = []
    all_fp = []
    all_fn = []
    total_ignored_errors = 0
    total_ignored_fps = 0
    total_anomalies = 0  # Count total anomalies
    total_ml_issues = 0  # Count total ML detection issues
    
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
        
        # Handle ML unified results
        ml_results = result.get('unified_results', [])
        total_ml_issues += len(ml_results)
        
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
        
        # List ML detection types and counts if we have ML results
        if total_ml_issues > 0:
            ml_types = {}
            for result in evaluation_results:
                for ml_issue in result.get('unified_results', []):
                    detection_type = ml_issue.get('detection_type', 'Unknown')
                    ml_types[detection_type] = ml_types.get(detection_type, 0) + 1
            
            summary_lines.append("\nML DETECTION TYPES:")
            for ml_type, count in ml_types.items():
                summary_lines.append(f"  - {ml_type}: {count}")
    
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
    
    full_results_path = os.path.join(os.path.dirname(__file__), output_dir, 'full_evaluation_results.json')
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
        
        serializable_results = make_serializable(evaluation_results)
        
        # Create enhanced output with human-readable summary at the top
        enhanced_output = {
            "human_readable_summary": performance_summary,
            "cell_coordinates": cell_coordinates,
            "detailed_evaluation_results": serializable_results
        }
        
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
    parser.add_argument("--run", choices=["validation", "anomaly", "ml", "both", "all"], default="both", help="Specify which analysis to run: validation, anomaly detection, ml detection, both (validation+anomaly), or all three.")
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
    parser.add_argument("--error-probability", type=float, default=0.1, help="Probability of injecting errors in each row (default: 0.1).")
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
    
    # If running "both", try to include all methods but keep it simple
    if args.run == "both":
        run_ml = False  # For now, don't auto-include ML with "both" to avoid complexity
    
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
            if run_validation or run_anomaly:
                print("Continuing without ML-based anomaly detection.")
                run_ml = False
            else:
                raise ConfigurationError(f"Could not initialize ML anomaly detector.\nDetails: {e}")
    
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
        field_mapper=field_mapper
    )
    
    # Run evaluation - Generate comprehensive samples with both errors and anomalies
    print(f"Generating {args.num_samples} samples with errors and anomalies...")
    
    all_samples = []
    
    # Load anomaly injection rules
    anomaly_rules = []
    try:
        anomaly_rules_path = os.path.join(os.path.dirname(__file__), 'anomaly_detectors', 'anomaly_injection_rules', f"{validator_name}.json")
        anomaly_rules = load_anomaly_rules(anomaly_rules_path)
        print(f"Loaded {len(anomaly_rules)} anomaly injection rules")
    except Exception:
        print(f"No anomaly injection rules available for {validator_name}")
    
    # Generate samples with both error injection and anomaly injection
    for i in range(args.num_samples):
        sample_df = df.copy()
        sample_df.name = f"sample_{i}"
        all_injected_items = []
        
        # Apply error injection (validation errors) if available
        if run_validation and 'rules' in locals():
            error_injector = ErrorInjector(rules, field_mapper=field_mapper)
            sample_df, error_injections = error_injector.inject_errors(
                sample_df, field_name, max_errors=args.max_errors//2, 
                error_probability=args.error_probability/2
            )
            all_injected_items.extend(error_injections)
        
        # Apply anomaly injection (semantic anomalies) if available
        if anomaly_rules:
            anomaly_injector = AnomalyInjector(anomaly_rules)
            sample_df, anomaly_injections = anomaly_injector.inject_anomalies(
                sample_df, field_name, max_anomalies=args.max_errors//2, 
                anomaly_probability=args.error_probability/2
            )
            all_injected_items.extend(anomaly_injections)
        
        # Save sample
        sample_path = os.path.join(os.path.dirname(__file__), args.output_dir, f"sample_{i}.csv")
        sample_df.to_csv(sample_path, index=False)
        
        # Save injection metadata
        injections_path = os.path.join(os.path.dirname(__file__), args.output_dir, f"sample_{i}_injected_items.json")
        with open(injections_path, 'w') as f:
            json.dump(all_injected_items, f, indent=2, ensure_ascii=False)
        
        all_samples.append({
            "data": sample_df, 
            "injected_errors": all_injected_items,  # Combined errors and anomalies
            "sample_index": i
        })
    
    error_samples = all_samples
    print(f"Generated samples with {sum(len(s['injected_errors']) for s in all_samples)} total injected items")
    
    # Run detection methods once on the full dataset
    print(f"Running detection methods on dataset with {len(df)} rows...")
    
    # Use unified approach when running multiple detection methods to avoid duplication
    use_unified = (run_ml and (run_validation or run_anomaly)) or (run_validation and run_anomaly and run_ml)
    
    # Run detection once on the original dataset
    dataset_results = evaluator.evaluate_sample(
        df, 
        field_name, 
        injected_errors=[],  # No injected errors for the base dataset
        run_validation=run_validation,
        run_anomaly_detection=run_anomaly,
        use_unified_approach=use_unified,
        validation_threshold=args.validation_threshold,
        anomaly_threshold=args.anomaly_threshold,
        ml_threshold=args.ml_threshold
    )
    
    print(f"Detection complete. Found:")
    if run_validation:
        print(f"  - Validation errors: {dataset_results.get('total_validation_errors', 0)}")
    if run_anomaly:
        print(f"  - Anomalies: {dataset_results.get('total_anomalies', 0)}")
    if run_ml:
        print(f"  - ML issues: {dataset_results.get('total_ml_issues', 0)}")
    
    # Now evaluate performance against each sample's injected errors
    evaluation_results = []
    for sample in error_samples:
        # Calculate performance metrics by comparing detection results with sample's injected errors
        sample_result = {
            "field_name": field_name,
            "row_count": len(df),
            "sample_id": sample.get("sample_index", f"sample_{len(evaluation_results)}"),
            "sample_path": f"sample_{sample.get('sample_index', len(evaluation_results))}.csv",
            
            # Copy detection results from the single dataset run
            "validation_results": dataset_results.get("validation_results", []),
            "anomaly_results": dataset_results.get("anomaly_results", []),
            "ml_results": dataset_results.get("ml_results", []),
            "total_validation_errors": dataset_results.get("total_validation_errors", 0),
            "total_anomalies": dataset_results.get("total_anomalies", 0),
            "total_ml_issues": dataset_results.get("total_ml_issues", 0),
            
            # Approach availability flags
            "validation_available": dataset_results.get("validation_available", False),
            "anomaly_detection_available": dataset_results.get("anomaly_detection_available", False),
            "ml_detection_available": dataset_results.get("ml_detection_available", False),
            "unified_approach_used": dataset_results.get("unified_approach_used", False),
        }
        
        # If we have unified results, copy them too
        if "unified_results" in dataset_results:
            sample_result.update({
                "unified_results": dataset_results["unified_results"],
                "unified_total_issues": dataset_results.get("unified_total_issues", 0),
                "unified_issues_by_type": dataset_results.get("unified_issues_by_type", {}),
            })
        
        # Calculate performance metrics for this sample's injected errors
        if sample.get("injected_errors") and run_validation:
            validation_results_for_metrics = sample_result["validation_results"]
            metrics = evaluator._calculate_metrics(
                sample["data"],  # Use the sample DataFrame with injected errors
                field_name, 
                validation_results_for_metrics, 
                sample["injected_errors"]
            )
            sample_result.update(metrics)
        
        # Add flags to indicate what was run
        if not run_validation:
            sample_result["validation_performed"] = False
            
        evaluation_results.append(sample_result)
    
    # Report results
    generate_summary_report(evaluation_results, args.output_dir, args.ignore_fp)


if __name__ == '__main__':
    main()
