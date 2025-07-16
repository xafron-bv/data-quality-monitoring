#!/usr/bin/env python3
"""
Script to analyze unique values in any column of a CSV file
"""

import pandas as pd
import sys
import argparse
from collections import Counter

def analyze_column_values(csv_file, column_name):
    """
    Analyze unique values in the specified column
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded CSV: {csv_file}")
        print(f"Total rows: {len(df)}")
        
        # Check if the column exists
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Get the specified column
        column_series = df[column_name]
        
        # Basic statistics
        print(f"\n📊 Basic Statistics for '{column_name}' column:")
        print(f"  - Total entries: {len(column_series)}")
        print(f"  - Non-null entries: {column_series.notna().sum()}")
        print(f"  - Null entries: {column_series.isna().sum()}")
        print(f"  - Unique values: {column_series.nunique()}")
        
        # Get unique values (excluding NaN)
        unique_values = column_series.dropna().unique()
        unique_values_sorted = sorted(unique_values)
        
        print(f"\n🔍 All Unique Values in '{column_name}' ({len(unique_values_sorted)} total):")
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
        
        # Save results to file
        output_file = f'{column_name.replace(" ", "_").lower()}_analysis.txt'
        with open(output_file, 'w') as f:
            f.write(f"{column_name} Analysis for {csv_file}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total unique values: {len(unique_values_sorted)}\n\n")
            
            f.write(f"All Unique Values in '{column_name}':\n")
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
        print("Usage: python analyze_colours.py <csv_file> [column_name]")
        print("  csv_file: Path to the CSV file to analyze")
        print("  column_name: Name of the column to analyze (default: 'colour_name')")
        print("Example: python analyze_colours.py data/esqualo_2022_fall_original.csv")
        print("Example: python analyze_colours.py data/esqualo_2022_fall_original.csv article_structure_name_2")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    column_name = sys.argv[2] if len(sys.argv) > 2 else 'colour_name'
    
    print(f"🔍 Analyzing column '{column_name}' in file: {csv_file}")
    
    unique_values = analyze_column_values(csv_file, column_name)
    
    if unique_values is not None:
        print(f"\n✅ Analysis complete! Found {len(unique_values)} unique values in '{column_name}' column.")

if __name__ == "__main__":
    main()
