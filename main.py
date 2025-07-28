#!/usr/bin/env python3
"""
Main entrypoint for the anomaly detection project.
This script provides a unified interface to run all project entrypoints.
"""

import argparse
import os
import sys

# Add current directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_column.analyze_column import entry as analyze_column_entry
from anomaly_detectors.llm_based.llm_model_training import entry as llm_train_entry

# Import entry functions from all entrypoints
from anomaly_detectors.ml_based.index import entry as ml_index_entry
from ml_curve_generator.ml_curve_generator import entry as ml_curves_entry
from multi_sample_evaluation.multi_sample_evaluation import main as multi_eval_entry
from single_sample_multi_field_demo.single_sample_multi_field_demo import main as single_demo_entry


def parse_args_to_dict(args_list):
    """Convert command line arguments to dictionary for entry functions."""
    result = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            # Check if this is a flag (no value)
            if i + 1 >= len(args_list) or args_list[i + 1].startswith('--'):
                result[key] = True
            else:
                result[key] = args_list[i + 1]
                i += 1
        else:
            # Positional argument
            if 'positional_args' not in result:
                result['positional_args'] = []
            result['positional_args'].append(arg)
        i += 1
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Main entrypoint for anomaly detection project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  ml-index          Generate ML model indexes
  llm-train         Train LLM models
  analyze-column    Analyze a specific column
  ml-curves         Generate ML performance curves
  single-demo       Run single sample demonstration
  multi-eval        Run multi-sample evaluation

Examples:
  python main.py ml-index data.csv
  python main.py llm-train data.csv --field color_name
  python main.py single-demo --brand esqualo --enable-all
  python main.py multi-eval --input data.csv --output results/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ML Index Generation
    ml_index_parser = subparsers.add_parser('ml-index', help='Generate ML model indexes')
    ml_index_parser.add_argument('args', nargs='*', help='Arguments to pass to index.py')

    # LLM Training
    llm_train_parser = subparsers.add_parser('llm-train', help='Train LLM models')
    llm_train_parser.add_argument('args', nargs='*', help='Arguments to pass to llm_model_training.py')

    # Column Analysis
    analyze_parser = subparsers.add_parser('analyze-column', help='Analyze a specific column')
    analyze_parser.add_argument('args', nargs='*', help='Arguments to pass to analyze_column.py')

    # ML Curves
    curves_parser = subparsers.add_parser('ml-curves', help='Generate ML performance curves')
    curves_parser.add_argument('args', nargs='*', help='Arguments to pass to ml_curve_generator.py')

    # Single Sample Demo
    single_parser = subparsers.add_parser('single-demo', help='Run single sample demonstration')
    single_parser.add_argument('args', nargs='*', help='Arguments to pass to single_sample_multi_field_demo.py')

    # Multi-Sample Evaluation
    multi_parser = subparsers.add_parser('multi-eval', help='Run multi-sample evaluation')
    multi_parser.add_argument('args', nargs='*', help='Arguments to pass to multi_sample_evaluation.py')

    # Special workflow commands
    workflow_parser = subparsers.add_parser('workflow', help='Run predefined workflows')
    workflow_parser.add_argument('type', choices=['train-all', 'test-all', 'demo'],
                                help='Workflow type to run')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Handle workflow commands
        if args.command == 'workflow':
            if args.type == 'train-all':
                # Train ML models (needs a CSV file)
                print("\n" + "="*60)
                print("ML Index Generation")
                print("="*60)
                print("\nNote: You need to provide a CSV file for training.")
                print("Example: python anomaly_detectors/ml_based/index.py data.csv")

                print("\n" + "="*60)
                print("LLM Model Training")
                print("="*60)
                print("\nNote: You need to provide a CSV file and field name.")
                print("Example: python anomaly_detectors/llm_based/llm_model_training.py data.csv --field color_name")

            elif args.type == 'test-all':
                # Just show help for all commands
                print("\n" + "="*60)
                print("Testing all entrypoints by showing their help")
                print("="*60)

                # These would need proper argument parsing to work
                print("\nColumn Analysis: python analyze_column/analyze_column.py --help")
                print("ML Curves: python ml_curve_generator/ml_curve_generator.py --help")
                print("Single Demo: python single_sample_multi_field_demo/single_sample_multi_field_demo.py --help")
                print("Multi Eval: python multi_sample_evaluation/multi_sample_evaluation.py --help")

            elif args.type == 'demo':
                # Run a basic demo
                print("\n" + "="*60)
                print("Running Single Sample Demo")
                print("="*60)
                # For single_demo, we need to simulate command-line args
                import sys
                old_argv = sys.argv
                try:
                    sys.argv = ['single_sample_multi_field_demo.py', '--brand', 'esqualo',
                               '--enable-validation', '--enable-pattern', '--enable-ml',
                               '--data-file', 'data/esqualo_2022_fall.csv']
                    single_demo_entry()
                finally:
                    sys.argv = old_argv

        else:
            # Parse arguments for individual commands
            parsed_args = parse_args_to_dict(args.args if hasattr(args, 'args') else [])

            if args.command == 'ml-index':
                # Extract positional csv_file if present
                if 'positional_args' in parsed_args and parsed_args['positional_args']:
                    csv_file = parsed_args['positional_args'][0]
                    del parsed_args['positional_args']
                    ml_index_entry(csv_file=csv_file, **parsed_args)
                else:
                    print("Error: csv_file is required for ml-index")
                    return 1

            elif args.command == 'llm-train':
                # Extract positional data_file if present
                if 'positional_args' in parsed_args and parsed_args['positional_args']:
                    data_file = parsed_args['positional_args'][0]
                    del parsed_args['positional_args']
                    llm_train_entry(data_file=data_file, **parsed_args)
                else:
                    print("Error: data_file is required for llm-train")
                    return 1

            elif args.command == 'analyze-column':
                # Extract positional csv_file and optional field_name
                if 'positional_args' in parsed_args and parsed_args['positional_args']:
                    csv_file = parsed_args['positional_args'][0]
                    field_name = parsed_args['positional_args'][1] if len(parsed_args['positional_args']) > 1 else 'color_name'
                    if 'positional_args' in parsed_args:
                        del parsed_args['positional_args']
                    analyze_column_entry(csv_file=csv_file, field_name=field_name, **parsed_args)
                else:
                    print("Error: csv_file is required for analyze-column")
                    return 1

            elif args.command == 'ml-curves':
                # Extract positional data_file if present
                if 'positional_args' in parsed_args and parsed_args['positional_args']:
                    data_file = parsed_args['positional_args'][0]
                    del parsed_args['positional_args']
                    ml_curves_entry(data_file=data_file, **parsed_args)
                else:
                    print("Error: data_file is required for ml-curves")
                    return 1

            elif args.command == 'single-demo':
                # single_demo_entry uses argparse, so we need to simulate command line
                import sys
                old_argv = sys.argv
                try:
                    new_argv = ['single_sample_multi_field_demo.py']
                    for key, value in parsed_args.items():
                        if key == 'positional_args':
                            continue
                        if value is True:
                            new_argv.append(f'--{key.replace("_", "-")}')
                        else:
                            new_argv.extend([f'--{key.replace("_", "-")}', str(value)])
                    sys.argv = new_argv
                    single_demo_entry()
                finally:
                    sys.argv = old_argv

            elif args.command == 'multi-eval':
                # multi_eval_entry also uses argparse
                import sys
                old_argv = sys.argv
                try:
                    new_argv = ['multi_sample_evaluation.py']
                    for key, value in parsed_args.items():
                        if key == 'positional_args':
                            continue
                        if value is True:
                            new_argv.append(f'--{key.replace("_", "-")}')
                        else:
                            new_argv.extend([f'--{key.replace("_", "-")}', str(value)])
                    sys.argv = new_argv
                    multi_eval_entry()
                finally:
                    sys.argv = old_argv

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
