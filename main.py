#!/usr/bin/env python3
"""
Main entrypoint for the anomaly detection project.
This script provides a unified interface to run all project entrypoints.
"""

import argparse
import json
import os
import sys

# Add current directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_column.analyze_column import entry as analyze_column_entry
from anomaly_detectors.llm_based.llm_model_training import entry as llm_train_entry

# Import entry functions from all entrypoints
from anomaly_detectors.ml_based.index import entry as ml_index_entry
from common.args_parser import create_parser_from_config, load_args_config
from ml_curve_generator.ml_curve_generator import entry as ml_curves_entry
from multi_sample_evaluation.multi_sample_evaluation import main as multi_eval_main
from single_sample_multi_field_demo.single_sample_multi_field_demo import main as single_demo_main

# Define entrypoint configurations
ENTRYPOINTS = {
    'ml-index': {
        'module': 'anomaly_detectors.ml_based',
        'entry': ml_index_entry,
        'args_json': 'anomaly_detectors/ml_based/args.json'
    },
    'llm-train': {
        'module': 'anomaly_detectors.llm_based',
        'entry': llm_train_entry,
        'args_json': 'anomaly_detectors/llm_based/args.json'
    },
    'analyze-column': {
        'module': 'analyze_column',
        'entry': analyze_column_entry,
        'args_json': 'analyze_column/args.json'
    },
    'ml-curves': {
        'module': 'ml_curve_generator',
        'entry': ml_curves_entry,
        'args_json': 'ml_curve_generator/args.json'
    },
    'single-demo': {
        'module': 'single_sample_multi_field_demo',
        'entry': single_demo_main,
        'args_json': 'single_sample_multi_field_demo/args.json',
        'uses_argparse': True  # This entry point parses args internally
    },
    'multi-eval': {
        'module': 'multi_sample_evaluation',
        'entry': multi_eval_main,
        'args_json': 'multi_sample_evaluation/args.json',
        'uses_argparse': True  # This entry point parses args internally
    }
}


def run_entrypoint(command: str, args: list):
    """Run a specific entrypoint with the given arguments."""
    if command not in ENTRYPOINTS:
        raise ValueError(f"Unknown command: {command}")

    config = ENTRYPOINTS[command]
    args_json_path = config['args_json']

    # Load the args configuration
    args_config = load_args_config(args_json_path)

    # If the entry point uses argparse internally, we need to set sys.argv
    if config.get('uses_argparse', False):
        import sys
        old_argv = sys.argv
        try:
            # Create a fake command line
            sys.argv = [f"{command}.py"] + args
            config['entry']()
        finally:
            sys.argv = old_argv
    else:
        # Parse arguments using the JSON config
        parser, positional_args = create_parser_from_config(args_config, prog_name=command)
        parsed_args = parser.parse_args(args)

        # Convert to dictionary
        args_dict = vars(parsed_args)

        # Call the entry function with unpacked arguments
        config['entry'](**args_dict)


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
  python main.py single-demo --brand esqualo --enable-validation --enable-ml
  python main.py multi-eval --input data.csv --output results/

For help on a specific command:
  python main.py <command> --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create subparsers for each command
    for cmd_name, cmd_config in ENTRYPOINTS.items():
        # Load the description from args.json
        try:
            args_config = load_args_config(cmd_config['args_json'])
            description = args_config.get('description', f'{cmd_name} command')
        except:
            description = f'{cmd_name} command'

        cmd_parser = subparsers.add_parser(cmd_name, help=description, add_help=False)
        cmd_parser.add_argument('args', nargs='*', help=f'Arguments for {cmd_name}')

    # Special workflow commands
    workflow_parser = subparsers.add_parser('workflow', help='Run predefined workflows')
    workflow_parser.add_argument('type', choices=['train-all', 'test-basic', 'demo'],
                                help='Workflow type to run')
    workflow_parser.add_argument('--data-file', help='Data file for workflows')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Handle workflow commands
        if args.command == 'workflow':
            if args.type == 'train-all':
                # Train ML models
                if not args.data_file:
                    print("Error: --data-file is required for train-all workflow")
                    return 1

                print("\n" + "="*60)
                print("ML Index Generation")
                print("="*60)
                run_entrypoint('ml-index', [args.data_file])

                print("\n" + "="*60)
                print("LLM Model Training")
                print("="*60)
                # Train for multiple fields
                for field in ['color_name', 'material', 'category']:
                    print(f"\nTraining LLM for field: {field}")
                    run_entrypoint('llm-train', [args.data_file, '--field', field])

            elif args.type == 'test-basic':
                # Test basic functionality
                print("\n" + "="*60)
                print("Testing Basic Functionality")
                print("="*60)

                # Show help for each command
                for cmd in ['ml-index', 'llm-train', 'analyze-column', 'ml-curves', 'single-demo', 'multi-eval']:
                    print(f"\n--- {cmd} help ---")
                    try:
                        run_entrypoint(cmd, ['--help'])
                    except SystemExit:
                        pass  # Ignore help exit

            elif args.type == 'demo':
                # Run a basic demo
                print("\n" + "="*60)
                print("Running Single Sample Demo")
                print("="*60)

                demo_args = [
                    '--brand', 'esqualo',
                    '--enable-validation', '--enable-pattern', '--enable-ml'
                ]

                if args.data_file:
                    demo_args.extend(['--data-file', args.data_file])
                else:
                    # Try to find a default data file
                    if os.path.exists('data/esqualo_2022_fall.csv'):
                        demo_args.extend(['--data-file', 'data/esqualo_2022_fall.csv'])

                run_entrypoint('single-demo', demo_args)

        else:
            # Run individual commands
            if hasattr(args, 'args'):
                run_entrypoint(args.command, args.args)
            else:
                run_entrypoint(args.command, [])

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
