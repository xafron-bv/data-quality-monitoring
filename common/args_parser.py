"""
Common argument parser utility for reading args.json configuration files.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def load_args_config(config_path: str) -> Dict[str, Any]:
    """Load arguments configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_parser_from_config(config: Dict[str, Any], prog_name: Optional[str] = None) -> argparse.ArgumentParser:
    """Create an ArgumentParser from a configuration dictionary."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=config.get('description', ''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Track positional arguments
    positional_args = []

    for arg_name, arg_config in config.get('arguments', {}).items():
        arg_type = arg_config.get('type', 'string')

        if arg_type == 'positional':
            # Positional argument
            kwargs = {
                'help': arg_config.get('help', '')
            }
            if 'default' in arg_config and not arg_config.get('required', False):
                kwargs['nargs'] = '?'
                kwargs['default'] = arg_config['default']
            parser.add_argument(arg_name, **kwargs)
            positional_args.append(arg_name)
        else:
            # Optional argument
            arg_names = [f'--{arg_name.replace("_", "-")}']
            kwargs = {
                'help': arg_config.get('help', ''),
                'default': arg_config.get('default')
            }

            if arg_type == 'flag':
                kwargs['action'] = 'store_true'
            elif arg_type == 'int':
                kwargs['type'] = int
            elif arg_type == 'float':
                kwargs['type'] = float
            elif arg_type == 'list':
                kwargs['nargs'] = '+'
                if 'item_type' in arg_config:
                    if arg_config['item_type'] == 'float':
                        kwargs['type'] = float
                    elif arg_config['item_type'] == 'int':
                        kwargs['type'] = int
            elif arg_type == 'string':
                kwargs['type'] = str

            if 'choices' in arg_config:
                kwargs['choices'] = arg_config['choices']

            if arg_config.get('required', False):
                kwargs['required'] = True

            parser.add_argument(*arg_names, **kwargs)

    return parser, positional_args


def parse_args_with_config(config_path: str, args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments using configuration from JSON file."""
    config = load_args_config(config_path)
    parser, _ = create_parser_from_config(config)
    return parser.parse_args(args)


def convert_args_to_dict(args: argparse.Namespace, positional_args: List[str]) -> Dict[str, Any]:
    """Convert argparse Namespace to dictionary, handling positional args separately."""
    args_dict = vars(args).copy()

    # Extract positional arguments
    positionals = {}
    for arg in positional_args:
        if arg in args_dict:
            positionals[arg] = args_dict.pop(arg)

    return args_dict, positionals
