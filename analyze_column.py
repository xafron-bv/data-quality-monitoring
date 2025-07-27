"""
Utility script to analyze unique values in a specific column of a CSV file.
"""

import pandas as pd
import sys
import os
import argparse
from field_mapper import FieldMapper
from exceptions import FileOperationError, DataError, ConfigurationError
from brand_configs import get_brand_config_manager

def analyze_field_values(csv_file, field_name, field_mapper=None):
    """
    Analyze unique values in the specified field
    """
    if field_mapper is None:
        field_mapper = FieldMapper.from_default_mapping()
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded CSV: {csv_file}")
        print(f"Total rows: {len(df)}")
        
        # Get the column name for the field using centralized mapping
        try:
            column_name = field_mapper.validate_column_exists(df, field_name)
        except ValueError as e:
            raise DataError(str(e)) from e
        
        # Get the specified column
        column_series = df[column_name]
        
        # Basic statistics
        print(f"\n📊 Basic Statistics for '{field_name}' field (column: '{column_name}'):")
        print(f"  - Total entries: {len(column_series)}")
        print(f"  - Non-null entries: {column_series.notna().sum()}")
        print(f"  - Null entries: {column_series.isna().sum()}")
        print(f"  - Unique values: {column_series.nunique()}")
        
        # Get unique values (excluding NaN)
        unique_values = column_series.dropna().unique()
        unique_values_sorted = sorted(unique_values)
        
        print(f"\n🔍 All Unique Values in '{field_name}' field ({len(unique_values_sorted)} total):")
        print("=" * 50)
        
        for i, value in enumerate(unique_values_sorted, 1):
            print(f"{i:2d}. {value}")
        
        # Value counts (frequency of each value)
        print(f"\n📈 Value Frequencies:")
        print("=" * 50)
        
        value_counts = column_series.value_counts()
        for value, count in value_counts.head(20).items():  # Top 20 most frequent
            print(f"{value:<30} : {count:>3} occurrences")
        
        if len(value_counts) > 20:
            print(f"... and {len(value_counts) - 20} more values")
        
        # Check for potential variations/duplicates
        print(f"\n🔍 Potential Variations Analysis:")
        print("=" * 50)
        
        # Group by lowercase to find case variations
        lowercase_groups = {}
        for value in unique_values_sorted:
            key = str(value).lower().strip()
            if key not in lowercase_groups:
                lowercase_groups[key] = []
            lowercase_groups[key].append(value)
        
        variations_found = False
        for key, variations in lowercase_groups.items():
            if len(variations) > 1:
                print(f"Case variations for '{key}': {variations}")
                variations_found = True
        
        if not variations_found:
            print("No case variations found.")
        
        # Check for whitespace variations
        print(f"\n🔍 Whitespace Variations:")
        print("=" * 50)
        
        whitespace_variations = []
        for value in unique_values_sorted:
            stripped = str(value).strip()
            if str(value) != stripped:
                whitespace_variations.append(f"'{value}' -> '{stripped}'")
        
        if whitespace_variations:
            for variation in whitespace_variations:
                print(variation)
        else:
            print("No whitespace variations found.")
        
        # Create analysis_results directory if it doesn't exist
        os.makedirs('analysis_results', exist_ok=True)
        
        # Save results to file
        output_file = f'analysis_results/{field_name.replace(" ", "_").lower()}_analysis.txt'
        with open(output_file, 'w') as f:
            f.write(f"{field_name} Analysis for {csv_file}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total unique values: {len(unique_values_sorted)}\n\n")
            
            f.write(f"All Unique Values in '{field_name}':\n")
            f.write("-" * 30 + "\n")
            for i, value in enumerate(unique_values_sorted, 1):
                f.write(f"{i:2d}. {value}\n")
            
            f.write(f"\nValue Frequencies:\n")
            f.write("-" * 30 + "\n")
            for value, count in value_counts.items():
                f.write(f"{value:<30} : {count:>3}\n")
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return unique_values_sorted
        
    except FileNotFoundError:
        raise FileOperationError(
            f"CSV file not found: {csv_file}",
            details={
                'file_path': csv_file,
                'suggestion': 'Check that the CSV file path is correct'
            }
        )
    except Exception as e:
        raise DataError(f"Error analyzing CSV: {e}") from e

def main():
    parser = argparse.ArgumentParser(description='Analyze a specific column in a CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('field_name', nargs='?', default='color_name', 
                       help='Name of the field to analyze (default: color_name)')
    parser.add_argument('--brand', required=True, help='Brand name for field mapping')
    
    args = parser.parse_args()
    
    # Set up brand configuration
    brand_manager = get_brand_config_manager()
    brand_manager.set_current_brand(args.brand)
    print(f"Using brand configuration: {args.brand}")
    
    csv_file = args.csv_file
    field_name = args.field_name
    
    print(f"🔍 Analyzing field '{field_name}' in file: {csv_file}")
    
    try:
        unique_values = analyze_field_values(csv_file, field_name)
        if unique_values is not None:
            print(f"\n✅ Analysis complete! Found {len(unique_values)} unique values in '{field_name}' field.")
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'details'):
            for key, value in e.details.items():
                print(f"  {key}: {value}")
        sys.exit(1)  # Keep this one sys.exit for CLI script behavior

if __name__ == "__main__":
    main()
