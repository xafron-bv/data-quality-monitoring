#!/usr/bin/env python3
"""
Utility script for managing brand configurations.

Usage:
    python manage_brands.py --list                          # List all available brands
    python manage_brands.py --create brand_name             # Create a new brand template
    python manage_brands.py --show brand_name               # Show brand configuration
    python manage_brands.py --validate brand_name           # Validate brand configuration
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from brand_configs import get_brand_config_manager, create_brand_config_template


def list_brands():
    """List all available brands."""
    manager = get_brand_config_manager()
    brands = manager.list_brands()
    
    if not brands:
        print("No brands configured yet.")
        print("Use --create to create a new brand configuration.")
        return
    
    print("Available brands:")
    for brand in sorted(brands):
        config = manager.get_brand(brand)
        if config:
            print(f"  - {brand}")
            if config.default_data_path:
                print(f"    Data: {config.default_data_path}")
            if config.enabled_fields:
                print(f"    Fields: {', '.join(config.enabled_fields[:5])}")
                if len(config.enabled_fields) > 5:
                    print(f"            ... and {len(config.enabled_fields) - 5} more")


def create_brand(brand_name: str):
    """Create a new brand configuration template."""
    output_file = os.path.join("brand_configs", f"{brand_name}.json")
    
    if os.path.exists(output_file):
        print(f"Error: Brand configuration already exists at {output_file}")
        print("Edit the existing file or choose a different brand name.")
        return
    
    create_brand_config_template(brand_name)
    print(f"\nNext steps:")
    print(f"1. Edit {output_file} to set the correct column mappings")
    print(f"2. Update the data paths to point to your brand's data files")
    print(f"3. Adjust the enabled_fields list based on your needs")
    print(f"4. Optionally customize the thresholds for your brand")
    print(f"\nNote: Validation and anomaly detection rules are global and will work")
    print(f"      automatically with your brand once column mappings are configured.")


def show_brand(brand_name: str):
    """Show detailed brand configuration."""
    manager = get_brand_config_manager()
    config = manager.get_brand(brand_name)
    
    if not config:
        print(f"Error: Brand '{brand_name}' not found.")
        print(f"Use --list to see available brands.")
        return
    
    print(f"\nBrand Configuration: {brand_name}")
    print("=" * 50)
    
    # Data paths
    print("\nData Paths:")
    print(f"  Default data: {config.default_data_path or 'Not configured'}")
    print(f"  Training data: {config.training_data_path or 'Not configured'}")
    print(f"  ML models: {config.ml_models_path or 'Not configured'}")
    
    # Field mappings
    print("\nField Mappings:")
    for field, column in sorted(config.field_mappings.items()):
        print(f"  {field:25} -> {column}")
    
    # Enabled fields
    if config.enabled_fields:
        print(f"\nEnabled Fields ({len(config.enabled_fields)}):")
        for field in config.enabled_fields:
            print(f"  - {field}")
    
    # Thresholds
    if config.custom_thresholds:
        print("\nCustom Thresholds:")
        for key, value in config.custom_thresholds.items():
            print(f"  {key}: {value}")


def validate_brand(brand_name: str):
    """Validate brand configuration."""
    manager = get_brand_config_manager()
    config = manager.get_brand(brand_name)
    
    if not config:
        print(f"Error: Brand '{brand_name}' not found.")
        return
    
    print(f"Validating brand configuration: {brand_name}")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check for placeholder values
    for field, column in config.field_mappings.items():
        if column.startswith("YOUR_"):
            issues.append(f"Field '{field}' has placeholder column name: {column}")
    
    # Check data paths
    if config.default_data_path:
        if not os.path.exists(config.default_data_path):
            warnings.append(f"Default data file not found: {config.default_data_path}")
    else:
        issues.append("No default data path configured")
    
    if config.training_data_path:
        if not os.path.exists(config.training_data_path):
            warnings.append(f"Training data file not found: {config.training_data_path}")
    
    # Check enabled fields
    if not config.enabled_fields:
        warnings.append("No enabled fields configured")
    else:
        for field in config.enabled_fields:
            if field not in config.field_mappings:
                issues.append(f"Enabled field '{field}' has no mapping configured")
    
    # Report results
    if issues:
        print(f"\n❌ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ Brand configuration is valid!")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage brand configurations for the data quality monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_brands.py --list                    # List all brands
  python manage_brands.py --create mybrand          # Create new brand template
  python manage_brands.py --show mybrand            # Show brand configuration
  python manage_brands.py --validate mybrand        # Validate brand configuration
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available brands")
    parser.add_argument("--create", metavar="BRAND_NAME", help="Create a new brand configuration template")
    parser.add_argument("--show", metavar="BRAND_NAME", help="Show detailed brand configuration")
    parser.add_argument("--validate", metavar="BRAND_NAME", help="Validate brand configuration")
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.list:
        list_brands()
    elif args.create:
        create_brand(args.create)
    elif args.show:
        show_brand(args.show)
    elif args.validate:
        validate_brand(args.validate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()