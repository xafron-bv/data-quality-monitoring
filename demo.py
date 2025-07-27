#!/usr/bin/env python3
"""
Data Quality Monitoring System Demo

This script demonstrates the capabilities of the data quality monitoring system
for detecting errors and anomalies in fashion product data using a comprehensive
single-sample approach. It showcases:

1. Comprehensive error and anomaly injection across ALL available fields
2. Three-tiered detection approach:
   - Validation (rule-based, high confidence)
   - Anomaly Detection (pattern-based, medium confidence)  
   - ML Detection (semantic similarity, adaptive)
3. Cell-level classification with priority: validation > pattern-based > ML-based
4. Consolidated reporting compatible with data_quality_viewer.html

The demo uses a single comprehensive sample with configurable injection intensity
to simulate real-world data quality issues across multiple fields simultaneously.
This approach is more realistic and efficient than the previous multi-sample method.
"""

import os
import sys
import json
import pandas as pd
import time
import argparse
from pathlib import Path

# Ensure we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import comprehensive detection modules
from comprehensive_sample_generator import generate_comprehensive_sample, save_comprehensive_sample
from comprehensive_detector import ComprehensiveFieldDetector
from consolidated_reporter import save_consolidated_reports
from confusion_matrix_analyzer import analyze_confusion_matrices
from common_interfaces import FieldMapper
from exceptions import DataQualityError, ConfigurationError, FileOperationError


class DataQualityDemo:
    """Comprehensive demo of the data quality monitoring system using single-sample approach."""
    
    def __init__(self, data_file: str = "data/esqualo_2022_fall.csv", 
                 output_dir: str = "demo_results",
                 injection_intensity=0.2, max_issues_per_row=2, core_fields_only=False,
                 enable_validation=True, enable_pattern=True, enable_ml=True, enable_llm=False,
                 llm_threshold=0.6, llm_few_shot_examples=False, 
                 llm_temporal_column=None, llm_context_columns=None):
        """
        Initialize the demo with the specified parameters.
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.injection_intensity = injection_intensity
        self.max_issues_per_row = max_issues_per_row
        self.core_fields_only = core_fields_only
        self.enable_validation = enable_validation
        self.enable_pattern = enable_pattern
        self.enable_ml = enable_ml
        self.enable_llm = enable_llm
        self.llm_threshold = llm_threshold
        self.llm_few_shot_examples = llm_few_shot_examples
        self.llm_temporal_column = llm_temporal_column
        self.llm_context_columns = llm_context_columns.split(',') if llm_context_columns else None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize field mapper
        self.field_mapper = FieldMapper.from_default_mapping()
        
        print(f"🔍 Data Quality Monitoring System Demo")
        print(f"📊 Target dataset: {self.data_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Injection intensity: {injection_intensity * 100:.1f}% of cells")
        print(f"🔧 Max issues per row: {max_issues_per_row}")
        print(f"🤖 ML detection: {'ENABLED' if enable_ml else 'DISABLED'}")
        print(f"🧠 LLM detection: {'ENABLED' if enable_llm else 'DISABLED'}")
        if enable_llm:
            print(f"   🎯 LLM threshold: {llm_threshold}")
            print(f"   📚 Few-shot examples: {'ENABLED' if llm_few_shot_examples else 'DISABLED'}")
            if llm_temporal_column:
                print(f"   ⏰ Temporal column: {llm_temporal_column}")
            if llm_context_columns:
                print(f"   🏷️  Context columns: {', '.join(llm_context_columns)}")
        print(f"📋 Fields: {'CORE ONLY' if core_fields_only else 'ALL AVAILABLE'} ({'material, color_name, category, size, care_instructions' if core_fields_only else 'all fields with detection capabilities'})")
        print("=" * 80)
    
    def setup_demo_environment(self):
        """Set up demo directories and load data."""
        print("🛠️  Setting up demo environment...")
        
        # Create demo output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and examine the data
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Check available fields for detection
            detector = ComprehensiveFieldDetector(field_mapper=self.field_mapper)
            available_detection_fields = detector.get_available_detection_fields()
            
            print(f"📋 Available fields for detection: {len(available_detection_fields)}")
            for field_name, capabilities in available_detection_fields.items():
                try:
                    column = self.field_mapper.validate_column_exists(self.df, field_name)
                    unique_values = self.df[column].nunique()
                    non_null_count = self.df[column].notna().sum()
                    methods = [method for method, available in capabilities.items() if available]
                    print(f"   📋 {field_name} -> {column}: {unique_values} unique values, {non_null_count} non-null rows")
                    print(f"      Detection methods: {', '.join(methods)}")
                except Exception as e:
                    print(f"   ❌ {field_name}: {e}")
            
            self.available_fields = list(available_detection_fields.keys())
            print(f"✅ Demo will analyze {len(self.available_fields)} fields")
            
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")
    
    def run_comprehensive_demo(self):
        """Run comprehensive demo using single-sample approach."""
        print(f"\n🎯 Running comprehensive data quality demo")
        print(f"   📍 Starting demo at {time.strftime('%H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Step 1: Generate comprehensive sample with errors and anomalies
            print(f"\n📋 Step 1: Generating comprehensive sample")
            sample_df, injection_metadata = generate_comprehensive_sample(
                df=self.df,
                injection_intensity=self.injection_intensity,
                max_issues_per_row=self.max_issues_per_row,
                field_mapper=self.field_mapper
            )
            
            # Save the comprehensive sample
            sample_files = save_comprehensive_sample(
                sample_df, injection_metadata, self.output_dir, "demo_sample"
            )
            print(f"   💾 Saved sample files: {list(sample_files.keys())}")
            
            # Step 2: Run comprehensive detection
            print(f"\n🔍 Step 2: Running comprehensive detection")
            detector = ComprehensiveFieldDetector(
                field_mapper=self.field_mapper,
                validation_threshold=0.0,  # Include all validation results
                anomaly_threshold=0.7,     # Default anomaly threshold
                ml_threshold=0.7,          # ML threshold
                llm_threshold=self.llm_threshold,  # LLM threshold
                batch_size=512,            # Smaller batch size to save memory
                max_workers=1,             # Single-threaded to prevent memory issues
                core_fields_only=self.core_fields_only,
                enable_validation=self.enable_validation,
                enable_pattern=self.enable_pattern,
                enable_ml=self.enable_ml,
                enable_llm=self.enable_llm
            )
            
            field_results, cell_classifications = detector.run_comprehensive_detection(sample_df)
            
            # Step 3: Generate consolidated reports
            print(f"\n📊 Step 3: Generating consolidated reports")
            report_files = save_consolidated_reports(
                cell_classifications=cell_classifications,
                field_results=field_results,
                sample_df=sample_df,
                output_dir=self.output_dir,
                injection_metadata=injection_metadata,
                sample_name="demo_analysis"
            )
            unified_report_path = report_files["unified_report"]
            
            # Step 4: Generate confusion matrix analysis
            print(f"\n📊 Step 4: Generating confusion matrix analysis")
            confusion_matrix_files = analyze_confusion_matrices(
                cell_classifications=cell_classifications,
                field_results=field_results,
                injection_metadata=injection_metadata,
                output_dir=self.output_dir,
                sample_name="demo_analysis",
                field_mapper=self.field_mapper,
                dataset_size=len(sample_df)
            )
            elapsed_time = time.time() - start_time
            print(f"\n✅ Demo completed successfully in {elapsed_time:.1f}s")
            print(f"\n📋 Demo Results Summary:")
            print(f"   📊 Dataset: {len(sample_df)} rows, {len(sample_df.columns)} columns")
            print(f"   🎯 Analyzed fields: {len(field_results)}")
            print(f"   🔍 Total issues detected: {len(cell_classifications)}")
            by_type = {}
            for classification in cell_classifications:
                detection_type = classification.detection_type
                by_type[detection_type] = by_type.get(detection_type, 0) + 1
            for detection_type, count in by_type.items():
                print(f"      {detection_type}: {count}")
            affected_rows = len(set(c.row_index for c in cell_classifications))
            print(f"   🎯 Affected rows: {affected_rows} / {len(sample_df)} ({affected_rows/len(sample_df)*100:.1f}%)")
            print(f"\n📁 Generated Unified Report:")
            print(f"   {unified_report_path}")
            
            return {
                "sample_files": sample_files,
                "report_files": report_files,
                "confusion_matrix_files": confusion_matrix_files,
                "field_results": field_results,
                "cell_classifications": cell_classifications,
                "injection_metadata": injection_metadata
            }
            
        except Exception as e:
            print(f"      ❌ Demo failed: {str(e)}")
            raise

    def run_complete_demo(self):
        """Run the complete demo workflow using comprehensive single-sample approach."""
        try:
            print("🚀 Starting Data Quality Monitoring System Demo")
            print("=" * 80)
            
            # Step 1: Setup
            self.setup_demo_environment()
            
            # Step 2: Run comprehensive demo
            demo_results = self.run_comprehensive_demo()
            
            print("\n🎉 Demo completed successfully!")
            print(f"📁 All results saved to: {self.output_dir}")
            
            print(f"\n🔑 KEY OUTPUT FILES:")
            print(f"   📄 Sample CSV: {os.path.basename(demo_results['sample_files']['sample_csv'])}")
            print(f"   📋 Unified Report: {os.path.basename(demo_results['report_files']['unified_report'])}")
            print(f"   📊 Confusion Matrix Report: {os.path.basename(demo_results['confusion_matrix_files']['json_report'])}")
            
            print(f"\n🔍 FOR VISUALIZATION:")
            print(f"   1. Open data_quality_viewer.html in your browser")
            print(f"   2. Upload CSV: {os.path.basename(demo_results['sample_files']['sample_csv'])}")
            print(f"   3. Upload JSON: {os.path.basename(demo_results['report_files']['viewer_report'])}")
            print(f"\n📊 Metrics Report: {os.path.basename(demo_results['report_files']['unified_report'])}")
            print(f"\n📈 Confusion Matrix Visualizations:")
            for viz_type, viz_path in demo_results['confusion_matrix_files'].items():
                if viz_type != 'json_report':
                    print(f"   • {viz_type.replace('_', ' ').title()}: {os.path.basename(viz_path)}")
            
            return demo_results
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            return None
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="Data Quality Monitoring System Demo - Comprehensive Single-Sample Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This demo generates a single comprehensive sample with configurable injection intensity
and runs all available detection methods (validation, pattern-based anomaly detection,
ML-based anomaly detection) across all fields simultaneously.

Example usage:
  python demo.py                                    # Default 20% injection intensity
  python demo.py --injection-intensity 0.1         # 10% injection intensity  
  python demo.py --injection-intensity 0.3 --max-issues-per-row 3  # Higher intensity
  python demo.py --data-file my_data.csv --output-dir my_results
        """
    )
    parser.add_argument("--data-file", default="data/esqualo_2022_fall.csv", 
                       help="Path to the CSV data file (default: data/esqualo_2022_fall.csv)")
    parser.add_argument("--output-dir", default="demo_results", 
                       help="Output directory for demo results (default: demo_results)")
    parser.add_argument("--injection-intensity", type=float, default=0.2,
                       help="Probability of injecting issues in each cell (0.0-1.0, default: 0.2)")
    parser.add_argument("--max-issues-per-row", type=int, default=2,
                       help="Maximum number of fields to corrupt per row (default: 2)")
    parser.add_argument("--validation-threshold", type=float, default=0.0,
                       help="Minimum confidence threshold for validation results (default: 0.0)")
    parser.add_argument("--anomaly-threshold", type=float, default=0.7,
                       help="Minimum confidence threshold for anomaly detection (default: 0.7)")
    parser.add_argument("--ml-threshold", type=float, default=0.7,
                       help="Minimum confidence threshold for ML detection (default: 0.7)")
    parser.add_argument("--llm-threshold", type=float, default=0.6,
                       help="Minimum confidence threshold for LLM detection (default: 0.6)")
    parser.add_argument("--core-fields-only", action="store_true",
                       help="Analyze only core fields (material, color_name, category, size, care_instructions) to save memory")
    parser.add_argument("--enable-validation", action="store_true", help="Enable validation (rule-based) detection")
    parser.add_argument("--enable-pattern", action="store_true", help="Enable pattern-based anomaly detection")
    parser.add_argument("--enable-ml", action="store_true", help="Enable ML-based anomaly detection")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM-based anomaly detection")
    parser.add_argument("--llm-few-shot-examples", action="store_true", 
                       help="Enable few-shot examples for LLM detection")
    parser.add_argument("--llm-temporal-column", type=str, default=None,
                       help="Column name containing temporal information for LLM dynamic encoding")
    parser.add_argument("--llm-context-columns", type=str, default=None,
                       help="Comma-separated list of context columns for LLM dynamic encoding")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 <= args.injection_intensity <= 1.0):
        print("❌ Error: injection-intensity must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.max_issues_per_row < 1:
        print("❌ Error: max-issues-per-row must be at least 1")
        sys.exit(1)
    
    # Create and run demo
    demo = DataQualityDemo(
        data_file=args.data_file,
        output_dir=args.output_dir,
        injection_intensity=args.injection_intensity,
        max_issues_per_row=args.max_issues_per_row,
        core_fields_only=args.core_fields_only,
        enable_validation=args.enable_validation,
        enable_pattern=args.enable_pattern,
        enable_ml=args.enable_ml,
        enable_llm=args.enable_llm,
        llm_threshold=args.llm_threshold,
        llm_few_shot_examples=args.llm_few_shot_examples,
        llm_temporal_column=args.llm_temporal_column,
        llm_context_columns=args.llm_context_columns
    )
    
    result = demo.run_complete_demo()
    
    if result:
        print("\n✅ Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
