import pandas as pd
import numpy as np
import random
import os
import json
import sys
import argparse
import string
import importlib
import re

# Add the script's directory to the Python path.
# This ensures that top-level modules like 'interfaces.py' are found
# when validator/reporter modules are loaded dynamically from subdirectories.
from validators.validation_error import ValidationError
from evaluator import Evaluator
from error_injection import apply_error_rule, generate_error_samples, load_error_rules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_module_class(module_path: str):
    """Dynamically loads a class from a Python file based on a module path string."""
    try:
        # e.g., 'validators.material.validate:MaterialValidator'
        module_name, class_name = module_path.split(':')
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error: Could not load class from '{module_path}'.\nEnsure the path is correct and the file contains the class.\nDetails: {e}")
        sys.exit(1)

def evaluate(validator, reporter, samples, column, ignore_errors=None, ignore_false_positives=False):
    """Evaluates the validator and reporter, returning a detailed analysis."""
    if ignore_errors is None:
        ignore_errors = []
        
    results = []
    for i, sample in enumerate(samples):
        df_with_errors = sample['data']
        injected_errors_info = sample['injected_errors']
        
        # Filter out errors that we've been asked to ignore
        injected_errors_to_consider = [
            e for e in injected_errors_info if e['error_rule'] not in ignore_errors
        ]
        
        # Run validator and reporter
        detected_by_validator = validator.bulk_validate(df_with_errors, column)
        reported_by_reporter = reporter.generate_report(detected_by_validator, df_with_errors)
        
        # Create a mapping of row_index to display_message from the reporter
        report_messages = {report['row_index']: report for report in reported_by_reporter}
        
        # Enhance validator results with reporter messages
        for error in detected_by_validator:
            row_idx = error.row_index
            if row_idx in report_messages:
                # We can't directly set attributes on ValidationError objects, so we'll
                # store the display message separately if needed later
                report_messages[row_idx]['validator_error'] = error

        detected_indices = {e.row_index for e in detected_by_validator}
        injected_indices = {e['row_index'] for e in injected_errors_to_consider}
        
        # --- Analysis ---
        true_positives = detected_indices.intersection(injected_indices)
        false_negatives = injected_indices - detected_indices # Injected but not detected
        
        # Only calculate false positives if we're not ignoring them
        if ignore_false_positives:
            false_positives = set()  # Empty set if we're ignoring false positives
            fp_details = []  # Empty list for false positive details
            ignored_fp_count = len(detected_indices - injected_indices)  # Count of ignored FPs
        else:
            false_positives = detected_indices - injected_indices
            fp_details = [e for e in detected_by_validator if e.row_index in false_positives]
            ignored_fp_count = 0

        results.append({
            "sample_index": i,
            "true_positives": true_positives,
            "false_negatives": [e for e in injected_errors_to_consider if e['row_index'] in false_negatives],
            "false_positives": fp_details,
            "ignored_error_count": len(injected_errors_info) - len(injected_errors_to_consider),
            "ignored_fp_count": ignored_fp_count
        })
        
    return results

def generate_summary_report(evaluation_results, output_dir, ignore_fp=False):
    """Generates a summary report of the evaluation results."""
    summary_lines = []
    summary_lines.append("=" * 80)
    
    # Determine if we have validation results or only anomaly detection
    has_validation = any('validation_performed' in result for result in evaluation_results)
    has_anomalies = any('anomaly_results' in result for result in evaluation_results)
    
    if has_validation and has_anomalies:
        summary_lines.append("VALIDATION AND ANOMALY DETECTION SUMMARY")
    elif has_validation:
        summary_lines.append("VALIDATION EVALUATION SUMMARY")
    else:
        summary_lines.append("ANOMALY DETECTION SUMMARY")
    
    summary_lines.append("=" * 80)
    
    all_tp = []
    all_fp = []
    all_fn = []
    total_ignored_errors = 0
    total_ignored_fps = 0
    total_anomalies = 0  # Count total anomalies
    
    for i, result in enumerate(evaluation_results):
        # Handle validation results if present
        if has_validation:
            true_positives = set(result.get('true_positives', []))
            false_positives = result.get('false_positives', [])
            false_negatives = result.get('false_negatives', [])
            
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
        
        sample_id = result.get('sample_id', f"Sample {i}")
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
        if has_validation:
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
    
    # Add false negatives section if validation was run
    if has_validation:
        # Only show false negatives section if we actually have validation results
        summary_lines.append(f"\n--- False Negatives (Missed Errors): {len(all_fn)} Total ---")
        if not all_fn:
            summary_lines.append("Excellent! No errors were missed.")
        else:
            fn_by_type = {}
            for fn in all_fn:
                rule = fn['error_rule']
                fn_by_type.setdefault(rule, []).append(fn['injected_data'])
            
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
                    code = fp.error_type
                    fp_by_type.setdefault(code, []).append(fp.error_data)

                for code, examples in fp_by_type.items():
                    summary_lines.append(f"\n  - Error Code: '{code}' (Flagged {len(examples)} times)")
                    for ex in list(set(examples))[:3]:
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

    # --- False Negatives (Missed Errors) ---
    summary_lines.append(f"\n--- False Negatives (Missed Errors): {len(all_fn)} Total ---")
    if not all_fn:
        summary_lines.append("Excellent! No errors were missed.")
    else:
        fn_by_type = {}
        for fn in all_fn:
            rule = fn['error_rule']
            fn_by_type.setdefault(rule, []).append(fn['injected_data'])
        
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
                code = fp.error_type
                fp_by_type.setdefault(code, []).append(fp.error_data)

            for code, examples in fp_by_type.items():
                summary_lines.append(f"\n  - Error Code: '{code}' (Flagged {len(examples)} times)")
                for ex in list(set(examples))[:3]:
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

    # Remove duplicated false negatives and false positives messages that happen due to the updated code
    # (this is a temporary fix until a more robust reporting system is implemented)
    text = "\n".join(summary_lines)
    lines = text.split("\n")
    unique_lines = []
    last_line = None
    
    for line in lines:
        if line != last_line:  # Skip duplicate consecutive lines
            unique_lines.append(line)
        last_line = line
    
    # Join back together
    summary_text = "\n".join(unique_lines)
    summary_text += "\n" + "="*80
    
    print(summary_text)

    # Save reports
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    full_results_path = os.path.join(output_dir, 'full_evaluation_results.json')
    with open(full_results_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        for result in evaluation_results:
            # Only convert true_positives to list if it exists (during validation runs)
            if 'true_positives' in result:
                result['true_positives'] = list(result['true_positives'])
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\nSummary report saved to '{summary_path}'")
    print(f"Full JSON results saved to '{full_results_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generic Validator/Reporter and Anomaly Detection Evaluation Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python evaluate.py data/source.csv --column material --output-dir results/material_test
  python evaluate.py data/source.csv --column "Care Instructions" --validator care_instructions
  python evaluate.py data/source.csv --column "colour_name" --anomaly-detector color_name --run anomaly

This will automatically look for:
- Rules:      rules/<validator>.json
- Validator:  validators/<validator>/validate.py (expecting class 'Validator')
- Reporter:   validators/report:Reporter
- Anomaly Detector: anomaly_detectors/<detector>/detect.py (expecting class 'AnomalyDetector')
- Anomaly Reporter: anomaly_detectors/report:AnomalyReporter

If --validator is not specified, it defaults to the value of --column (converted to snake_case if needed).
If --anomaly-detector is not specified, it defaults to the value of --validator.
"""
    )
    parser.add_argument("source_data", help="Path to the source CSV data file.")
    parser.add_argument("--column", required=True, help="The target column to validate in the CSV file (e.g., 'material', 'Care Instructions').")
    parser.add_argument("--validator", help="The validator name to use for files and classes (e.g., 'material', 'care_instructions'). Defaults to the column name if not provided.")
    parser.add_argument("--anomaly-detector", help="The anomaly detector name to use (e.g., 'material', 'color_name'). Defaults to the validator name if not provided.")
    parser.add_argument("--run", choices=["validation", "anomaly", "both"], default="both", help="Specify which analysis to run: validation, anomaly detection, or both.")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate for evaluation (default: 32).")
    parser.add_argument("--max-errors", type=int, default=3, help="Maximum number of errors to combine in a single sample (default: 3).")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save all evaluation results and generated samples.")
    parser.add_argument("--ignore-errors", nargs='+', default=[], help="A list of error rule names to ignore during evaluation (e.g., inject_unicode_error).")
    parser.add_argument("--ignore-fp", action="store_true", help="If set, false positives will be ignored in the evaluation.")
    args = parser.parse_args()

    # --- Derive paths and class names ---
    column_name = args.column
    
    # If validator name is not provided, default to column name (converted to snake_case)
    if args.validator:
        validator_name = args.validator
    else:
        # Convert column name to snake_case if needed (e.g., "Care Instructions" -> "care_instructions")
        validator_name = column_name.lower().replace(' ', '_')
    
    # If anomaly detector name is not provided, default to validator name
    if args.anomaly_detector:
        detector_name = args.anomaly_detector
    else:
        detector_name = validator_name
    
    # Determine what to run based on the --run argument
    run_validation = args.run in ["validation", "both"]
    run_anomaly = args.run in ["anomaly", "both"]
    
    rules_path = f"rules/{validator_name}.json"
    validator_module_str = f"validators.{validator_name}.validate:Validator"
    validator_reporter_module_str = f"validators.report:Reporter"
    anomaly_detector_module_str = f"anomaly_detectors.{detector_name}.detect:AnomalyDetector"
    anomaly_reporter_module_str = f"anomaly_detectors.report:AnomalyReporter"
    
    print(f"--- Evaluation Setup ---")
    print(f"Target column: '{column_name}'")
    
    if run_validation:
        print(f"Using validator: '{validator_name}'")
        print(f"Rules: {rules_path}")
        print(f"Validator: {validator_module_str}")
        print(f"Validator Reporter: {validator_reporter_module_str}")
    
    if run_anomaly:
        print(f"Using anomaly detector: '{detector_name}'")
        print(f"Anomaly Detector: {anomaly_detector_module_str}")
        print(f"Anomaly Reporter: {anomaly_reporter_module_str}")
        
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load files and modules
    try:
        df = pd.read_csv(args.source_data)
    except FileNotFoundError:
        print(f"Error: Source data file not found at '{args.source_data}'")
        sys.exit(1)

    validator = None
    validator_reporter = None
    anomaly_detector = None
    anomaly_reporter = None

    # Load validation components if needed
    if run_validation:
        try:
            rules = load_error_rules(rules_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Could not read or parse rules file at '{rules_path}'.\nDetails: {e}")
            sys.exit(1)

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
            if run_validation:
                print("Continuing with validation only.")
                run_anomaly = False
            else:
                sys.exit(1)
    
    # Create evaluator with appropriate components
    evaluator = Evaluator(
        validator=validator,
        validator_reporter=validator_reporter,
        anomaly_detector=anomaly_detector,
        anomaly_reporter=anomaly_reporter
    )
    
    # Run evaluation
    if run_validation:
        error_samples = generate_error_samples(df, column_name, rules, args.num_samples, args.max_errors, args.output_dir)
    else:
        # If only running anomaly detection, still need sample dataframes
        error_samples = []
        for i in range(args.num_samples):
            sample_df = df.copy()
            sample_df.name = f"sample_{i}"
            sample_path = os.path.join(args.output_dir, f"sample_{i}.csv")
            sample_df.to_csv(sample_path, index=False)
            error_samples.append({"data": sample_df, "injected_errors": [], "sample_index": i})
    
    evaluation_results = []
    for sample in error_samples:
        result = evaluator.evaluate_sample(
            sample["sample_df"], 
            column_name, 
            sample.get("injected_errors", []),
            run_validation=run_validation,
            run_anomaly_detection=run_anomaly
        )
        
        # Add flags to indicate what was run
        if not run_validation:
            result["validation_performed"] = False
        
        evaluation_results.append(result)
    
    # Report results
    generate_summary_report(evaluation_results, args.output_dir, args.ignore_fp)


if __name__ == '__main__':
    main()
