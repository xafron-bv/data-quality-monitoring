#!/usr/bin/env python3
"""
Script to analyze unique values in any field of a CSV file
"""

import pandas as pd
import sys
from field_column_map import get_field_to_column_map

def analyze_field_values(csv_file, field_name):
    """
    Analyze unique values in the specified field
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded CSV: {csv_file}")
        print(f"Total rows: {len(df)}")
        
        # Get the column name for the field
        field_to_column_map = get_field_to_column_map()
        column_name = field_to_column_map.get(field_name, field_name)
        
        # Check if the column exists
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' (mapped from field '{field_name}') not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
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
        import os
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
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_column.py <csv_file> [field_name]")
        print("  csv_file: Path to the CSV file to analyze")
        print("  field_name: Name of the field to analyze (default: 'color_name')")
        print("Example: python analyze_column.py data/esqualo_2022_fall_original.csv")
        print("Example: python analyze_column.py data/esqualo_2022_fall_original.csv category")
        print("Example: python analyze_column.py data/esqualo_2022_fall_original.csv material")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    field_name = sys.argv[2] if len(sys.argv) > 2 else 'color_name'
    
    print(f"🔍 Analyzing field '{field_name}' in file: {csv_file}")
    
    unique_values = analyze_field_values(csv_file, field_name)
    
    if unique_values is not None:
        print(f"\n✅ Analysis complete! Found {len(unique_values)} unique values in '{field_name}' field.")

if __name__ == "__main__":
    main()
