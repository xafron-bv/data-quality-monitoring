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

def apply_error_rule(data_string: str, rule: dict):
    """Applies a single error rule to a string."""
    if not isinstance(data_string, str):
        return data_string

    # Check conditions first
    if "conditions" in rule:
        for cond in rule["conditions"]:
            if cond["type"] == "contains" and cond["value"] not in data_string:
                return data_string # Condition not met, don't apply rule

    op = rule["operation"]
    params = rule.get("params", {})

    if op == "string_replace":
        return data_string.replace(params["find"], params["replace"])
    elif op == "regex_replace":
        count = params.get("count", 0) # Replace all if count is 0
        return re.sub(params["pattern"], params["replace"], data_string, count=count)
    elif op == "prepend":
        return params["text"] + data_string
    elif op == "append":
        return data_string + params["text"]
    elif op == "add_whitespace":
        return " " + data_string + " "
    elif op == "random_noise":
        noise_type = random.choice(['chars', 'duplicate'])
        if noise_type == 'chars':
            noise = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
            pos = random.randint(0, len(data_string))
            return data_string[:pos] + noise + data_string[pos:]
        else: # duplicate
            parts = data_string.split()
            if not parts: return data_string
            return data_string + " " + random.choice(parts)
    return data_string


def generate_error_samples(df: pd.DataFrame, column: str, rules: list, num_samples: int, max_errors_per_sample: int, output_dir: str):
    """Generates samples with errors based on the rules JSON and saves them."""
    samples = []
    print(f"Generating {num_samples} samples in '{output_dir}' with up to {max_errors_per_sample} errors each...")

    for i in range(num_samples):
        df_copy = df.copy()
        injected_errors = []
        
        num_to_inject = random.randint(1, max_errors_per_sample)
        k_sample = min(len(df.index), num_to_inject * 5)
        error_rows_indices = random.sample(list(df.index), k=k_sample)
        
        injected_count = 0
        for idx in error_rows_indices:
            if injected_count >= num_to_inject:
                break
            
            rule = random.choice(rules)
            original_data = df_copy.at[idx, column]
            
            new_data = apply_error_rule(original_data, rule)

            if new_data != original_data:
                df_copy.at[idx, column] = new_data
                injected_errors.append({
                    "row_index": idx,
                    "original_data": original_data,
                    "injected_data": new_data,
                    "error_rule": rule['rule_name']
                })
                injected_count += 1
        
        # Save the generated sample and its error log
        sample_csv_path = os.path.join(output_dir, f'sample_{i}.csv')
        injected_errors_path = os.path.join(output_dir, f'sample_{i}_injected_errors.json')
        
        df_copy.to_csv(sample_csv_path, index=False)
        with open(injected_errors_path, 'w') as f:
            json.dump(injected_errors, f, indent=4)

        samples.append({"data": df_copy, "injected_errors": injected_errors})
    print("Sample generation complete.")
    return samples

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

        detected_indices = {e['row_index'] for e in detected_by_validator}
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
            fp_details = [e for e in detected_by_validator if e['row_index'] in false_positives]
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

def generate_summary_report(evaluation_results, output_dir, ignore_fp=False, error_messages_path=None):
    """Generates and prints a summary, saving full results to the output directory. Also checks for missing error messages."""
    all_fn = [error for res in evaluation_results for error in res['false_negatives']]
    all_fp = [error for res in evaluation_results for error in res['false_positives']]
    total_ignored_errors = sum(res.get('ignored_error_count', 0) for res in evaluation_results)
    total_ignored_fps = sum(res.get('ignored_fp_count', 0) for res in evaluation_results)

    summary_lines = ["="*80, "EVALUATION SUMMARY REPORT", "="*80]
    
    # --- Summary stats ---
    summary_lines.append(f"\n--- Summary Statistics ---")
    summary_lines.append(f"  - Total errors ignored: {total_ignored_errors}")
    summary_lines.append(f"  - Total false positives ignored: {total_ignored_fps}")

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
                code = fp['error_code']
                fp_by_type.setdefault(code, []).append(fp['error_data'])

            for code, examples in fp_by_type.items():
                summary_lines.append(f"\n  - Error Code: '{code}' (Flagged {len(examples)} times)")
                for ex in list(set(examples))[:3]:
                    summary_lines.append(f"    - Example: '{ex}'")

    # --- Check for missing error messages ---
    if error_messages_path:
        try:
            with open(error_messages_path, 'r') as f:
                error_messages = json.load(f)
        except Exception as e:
            error_messages = {}
            summary_lines.append(f"\n[WARNING] Could not load error messages from {error_messages_path}: {e}")
        # Collect all error codes from validator output
        all_error_codes = set()
        for res in evaluation_results:
            for fp in res.get('false_positives', []):
                if 'error_code' in fp:
                    all_error_codes.add(fp['error_code'])
            for fn in res.get('false_negatives', []):
                if 'error_rule' in fn:
                    all_error_codes.add(fn['error_rule'])
        # Also check for error codes in true positives
        for res in evaluation_results:
            for tp_idx in res.get('true_positives', []):
                # Not enough info to get error_code for true positives, so skip
                pass
        # Check for missing error messages
        missing = [code for code in all_error_codes if code not in error_messages]
        if missing:
            summary_lines.append("\n[WARNING] The following error codes are missing human-readable messages in the error_messages.json file:")
            for code in missing:
                summary_lines.append(f"  - {code}")
        else:
            summary_lines.append("\nAll error codes have human-readable messages defined.")

    summary_lines.append("\n" + "="*80)
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save reports
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    full_results_path = os.path.join(output_dir, 'full_evaluation_results.json')
    with open(full_results_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        for result in evaluation_results:
            result['true_positives'] = list(result['true_positives'])
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\nSummary report saved to '{summary_path}'")
    print(f"Full JSON results saved to '{full_results_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generic Validator/Reporter Evaluation Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python evaluate.py data/source.csv material --output-dir results/material_test

This will automatically look for:
- Rules:      rules/material.json
- Validator:  validators/material/validate.py (expecting class 'MaterialValidator')
- Reporter:   validators/material/report.py (expecting class 'MaterialReporter')
"""
    )
    parser.add_argument("source_data", help="Path to the source CSV data file.")
    parser.add_argument("--column", help="The target column to validate (e.g., 'material', 'color'). This determines which rules and modules to load.")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate for evaluation (default: 32).")
    parser.add_argument("--max-errors", type=int, default=3, help="Maximum number of errors to combine in a single sample (default: 3).")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save all evaluation results and generated samples.")
    parser.add_argument("--ignore-errors", nargs='+', default=[], help="A list of error rule names to ignore during evaluation (e.g., inject_unicode_error).")
    parser.add_argument("--ignore-fp", action="store_true", help="If set, false positives will be ignored in the evaluation.")
    args = parser.parse_args()

    # --- Derive paths and class names from the 'column' argument ---
    column_name = args.column
    capitalized_name = column_name.capitalize()

    rules_path = f"rules/{column_name}.json"
    validator_module_str = f"validators.{column_name}.validate:{capitalized_name}Validator"
    reporter_module_str = f"validators.report:{capitalized_name}Reporter"
    
    print(f"--- Loading assets for column: '{column_name}' ---")
    print(f"Rules: {rules_path}")
    print(f"Validator: {validator_module_str}")
    print(f"Reporter: {reporter_module_str}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load files and modules
    try:
        df = pd.read_csv(args.source_data)
    except FileNotFoundError:
        print(f"Error: Source data file not found at '{args.source_data}'")
        sys.exit(1)

    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)['error_rules']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or parse rules file at '{rules_path}'.\nDetails: {e}")
        sys.exit(1)

    ValidatorClass = load_module_class(validator_module_str)
    ReporterClass = load_module_class(reporter_module_str)
    
    validator = ValidatorClass()
    reporter = ReporterClass()
    
    # Run evaluation
    error_samples = generate_error_samples(df, args.column, rules, args.num_samples, args.max_errors, args.output_dir)
    evaluation_results = evaluate(validator, reporter, error_samples, args.column, 
                                args.ignore_errors, args.ignore_fp)
    
    # Derive error_messages_path
    error_messages_path = f"validators/{column_name}/error_messages.json"
    # Report results
    generate_summary_report(evaluation_results, args.output_dir, args.ignore_fp, error_messages_path)


if __name__ == '__main__':
    main()
