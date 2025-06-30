import pandas as pd
import numpy as np
import random
import os
import json
import sys
import argparse
from material_validator import MaterialValidator

def inject_unicode_error(material):
    """Injects a unicode error into the material string."""
    if isinstance(material, str):
        return material.replace('é', 'Ãª')
    return material

def inject_composition_error(material):
    """Changes a composition so it doesn't add up to 100."""
    if isinstance(material, str):
        return material.replace('100%', '99%')
    return material

def inject_percentage_sign_error(material):
    """Removes a percentage sign."""
    if isinstance(material, str):
        return material.replace('%', '')
    return material

def inject_trademark_error(material):
    """Adds a trademark symbol."""
    if isinstance(material, str) and 'Cotton' in material:
        return material.replace('Cotton', 'Cotton™')
    return material

def inject_delimiter_error(material):
    """Changes a delimiter."""
    if isinstance(material, str):
        return material.replace(',', ';')
    return material

def inject_spacing_error(material):
    """Adds leading/trailing spaces."""
    if isinstance(material, str):
        return " " + material + " "
    return material

def inject_care_instruction_error(material):
    """Adds care instructions."""
    if isinstance(material, str):
        return material + " - Machine wash cold"
    return material

def inject_color_error(material):
    """Adds a color attribute."""
    if isinstance(material, str):
        return "Blue " + material
    return material

def inject_missing_composition_error(material):
    """Removes the composition for a part."""
    if isinstance(material, str) and 'Cotton' in material:
        return 'Cotton'
    return material

def inject_decimal_error(material):
    """Adds a decimal to a percentage."""
    if isinstance(material, str):
        return material.replace('100', '99.9')
    return material
    
def inject_format_error(material):
    """Changes the format of the material string."""
    if isinstance(material, str) and 'Cotton' in material and '100%' in material:
        return 'cotton 100'
    return material

def inject_hyphen_delimiter_error(material):
    """Replaces a space with a hyphen, e.g., '100% Cotton' -> '100%-Cotton'."""
    if isinstance(material, str):
        return material.replace('% ', '%-')
    return material

def inject_multiple_spaces_error(material):
    """Adds multiple spaces between words."""
    if isinstance(material, str):
        return material.replace(' ', '   ')
    return material

def inject_line_break_error(material):
    """Adds a line break in the middle of the string."""
    if isinstance(material, str) and ' ' in material:
        return material.replace(' ', '\n', 1)
    return material

ERROR_FUNCTIONS = [
    inject_unicode_error, inject_composition_error, inject_percentage_sign_error,
    inject_trademark_error, inject_delimiter_error, inject_spacing_error,
    inject_care_instruction_error, inject_color_error, inject_missing_composition_error,
    inject_decimal_error, inject_format_error, inject_hyphen_delimiter_error,
    inject_multiple_spaces_error, inject_line_break_error
]

def generate_error_samples(df, num_samples=32, max_errors=3, cache_dir='cache'):
    """
    Generates samples with errors, saving them to a cache directory.
    
    Args:
        df: DataFrame containing the data to inject errors into
        num_samples: Number of samples to generate
        max_errors: Maximum number of errors to combine in a single sample
        cache_dir: Directory to save the samples to
    """
    samples = []
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Generating {num_samples} samples with up to {max_errors} errors per sample and saving to '{cache_dir}/' directory...")

    for i in range(num_samples):
        df_copy = df.copy()
        injected_errors = []
        
        num_errors_to_inject = random.randint(1, max_errors) 
        error_rows_indices = random.sample(range(len(df_copy)), k=min(len(df_copy), 20)) 
        
        for idx in error_rows_indices:
            error_func = random.choice(ERROR_FUNCTIONS)
            original_material = df_copy.at[idx, 'material']
            
            if isinstance(original_material, str):
                new_material = error_func(original_material)
                if new_material != original_material:
                    df_copy.at[idx, 'material'] = new_material
                    injected_errors.append({
                        "row_index": idx,
                        "original_material": original_material,
                        "injected_material": new_material,
                        "error_type": error_func.__name__
                    })

        sample_csv_path = os.path.join(cache_dir, f'sample_{i}.csv')
        injected_errors_path = os.path.join(cache_dir, f'sample_{i}_injected_errors.json')
        
        df_copy.to_csv(sample_csv_path, index=False)
        with open(injected_errors_path, 'w') as f:
            json.dump(injected_errors, f, indent=4)

        samples.append({"data": df_copy, "injected_errors": injected_errors})
    print("Sample generation complete.")
    return samples

def evaluate(validator, samples, ignore_errors=None):
    """Evaluates the validator's performance, ignoring specified error types."""
    if ignore_errors is None:
        ignore_errors = []
    
    results = []
    for i, sample in enumerate(samples):
        df_with_errors = sample['data']
        injected_errors_info = sample['injected_errors']
        
        # Filter out the errors that we've been told to ignore for this evaluation run
        injected_errors_to_consider = [
            e for e in injected_errors_info if e['error_type'] not in ignore_errors
        ]
        
        detected_errors = validator.bulk_validate(df_with_errors)
        
        detected_indices = {e['row_index'] for e in detected_errors}
        injected_indices = {e['row_index'] for e in injected_errors_to_consider}
        
        true_positives = len(detected_indices.intersection(injected_indices))
        false_negatives = len(injected_indices - detected_indices)
        false_positives = len(detected_indices - injected_indices)
        
        total_injected = len(injected_indices)
        
        accuracy = (true_positives / total_injected) * 100 if total_injected > 0 else 100

        undetected_errors = [e for e in injected_errors_to_consider if e['row_index'] not in detected_indices]

        results.append({
            "sample_index": i,
            "total_injected_errors_considered": total_injected,
            "total_injected_errors_original": len(injected_errors_info),
            "ignored_error_count": len(injected_errors_info) - total_injected,
            "detected_errors": len(detected_errors),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives (undetected)": false_negatives,
            "accuracy_percentage": accuracy,
            "undetected_errors": undetected_errors,
        })
        
    return results

def generate_undetected_error_summary(evaluation_results, output_path='undetected_error_summary.txt'):
    """Generates a summary of undetected errors, grouped by type."""
    error_definitions = {
        func.__name__: func.__doc__.strip() for func in ERROR_FUNCTIONS
    }
    
    undetected_by_type = {}
    for result in evaluation_results:
        for error in result['undetected_errors']:
            error_type = error['error_type']
            if error_type not in undetected_by_type:
                undetected_by_type[error_type] = []
            undetected_by_type[error_type].append(error['injected_material'])

    summary_lines = ["--- Summary of Undetected Errors ---"]
    if not undetected_by_type:
        summary_lines.append("No errors were missed by the validator. Great job!")
    else:
        for error_type, examples in undetected_by_type.items():
            definition = error_definitions.get(error_type, "No definition found.")
            summary_lines.append(f"\n- Error Type: {error_type}")
            summary_lines.append(f"  - Definition: {definition}")
            summary_lines.append(f"  - Count: {len(examples)}")
            summary_lines.append("  - Examples:")
            unique_examples = list(set(examples))
            for ex in unique_examples[:3]:
                summary_lines.append(f"    - '{ex}'")

    summary = "\n".join(summary_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
        
    print("\n" + summary)
    print(f"\nSummary of undetected errors saved to '{output_path}'")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the Material Validator script.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument(
        "--ignore-errors",
        nargs='+',
        default=[],
        help="A list of error function names to ignore during evaluation (e.g., inject_spacing_error inject_multiple_spaces_error)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of samples to generate for evaluation (default: 32)."
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=3,
        help="Maximum number of errors to combine in a single sample (default: 3)."
    )
    args = parser.parse_args()
        
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: '{args.input_file}' not found.")
        return

    error_samples = generate_error_samples(df, num_samples=args.num_samples, max_errors=args.max_errors)
    validator = MaterialValidator()
    evaluation_results = evaluate(validator, error_samples, ignore_errors=args.ignore_errors)
    
    results_path = 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nEvaluation results saved to '{results_path}'")

    for result in evaluation_results:
        print(f"--- Sample {result['sample_index']} ---")
        print(f"  - Accuracy: {result['accuracy_percentage']:.2f}%")
        
    total_injected = sum(r['total_injected_errors_considered'] for r in evaluation_results)
    total_tp = sum(r['true_positives'] for r in evaluation_results)
    overall_accuracy = (total_tp / total_injected) * 100 if total_injected > 0 else 100
    
    print("\n--- Overall Performance ---")
    if args.ignore_errors:
        print(f"Ignoring the following error types: {', '.join(args.ignore_errors)}")
    print(f"Using {args.num_samples} samples with up to {args.max_errors} combined errors per sample")
    print(f"Total injected errors considered: {total_injected}")
    print(f"Total correctly detected errors: {total_tp}")
    print(f"Overall detection accuracy: {overall_accuracy:.2f}%")
    
    generate_undetected_error_summary(evaluation_results)


if __name__ == '__main__':
    main()
