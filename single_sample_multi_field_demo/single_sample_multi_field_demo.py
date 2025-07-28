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

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import comprehensive detection modules
from comprehensive_sample_generator import generate_comprehensive_sample, save_comprehensive_sample
from confusion_matrix_analyzer import analyze_confusion_matrices
from consolidated_reporter import save_consolidated_reports

# Import weight generation functions
from generate_detection_weights import generate_weights_from_performance, generate_weights_report, load_performance_data

from brand_config import get_available_brands, load_brand_config
from common.comprehensive_detector import ComprehensiveFieldDetector
from common.exceptions import ConfigurationError, DataQualityError, FileOperationError
from common.field_mapper import FieldMapper


class DataQualityDemo:
    """Comprehensive demo of the data quality monitoring system using single-sample approach."""

    def __init__(self, data_file: str, brand_name: str,
                 output_dir: str = "demo_results",
                 injection_intensity=0.2, max_issues_per_row=2, core_fields_only=False,
                 enable_validation=True, enable_pattern=True, enable_ml=True, enable_llm=False,
                 llm_threshold=0.6, llm_few_shot_examples=False,
                 llm_temporal_column=None, llm_context_columns=None, use_weighted_combination=False,
                 weights_file="detection_weights.json"):
        """
        Initialize the demo with the specified parameters.
        """
        self.data_file = data_file
        self.brand_name = brand_name
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
        self.use_weighted_combination = use_weighted_combination
        self.weights_file = weights_file

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize field mapper
        self.field_mapper = FieldMapper.from_brand(brand_name)

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
        print(f"🎯 Combination method: {'WEIGHTED' if use_weighted_combination else 'PRIORITY-BASED'}")
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
                enable_llm=self.enable_llm,
                use_weighted_combination=self.use_weighted_combination,
                weights_file=self.weights_file
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

    def generate_detection_weights(self, unified_report_path: str, output_file: str,
                                 baseline_weight: float = 0.1, verbose: bool = False) -> bool:
        """
        Generate detection weights from demo results.

        Args:
            unified_report_path: Path to the unified report JSON file
            output_file: Path to save the generated weights
            baseline_weight: Minimum weight for untrained methods
            verbose: Whether to print detailed information

        Returns:
            True if weights were generated successfully, False otherwise
        """
        try:
            print(f"\n🔧 Step 4: Generating detection weights")
            print(f"📊 Loading performance data from: {os.path.basename(unified_report_path)}")

            # Load performance data from unified report
            performance_data = load_performance_data(unified_report_path)

            if not performance_data:
                print(f"❌ Error: No field performance data found in unified report")
                return False

            print(f"✅ Found performance data for {len(performance_data)} fields")

            # Generate weights
            print(f"🔧 Generating weights with baseline weight: {baseline_weight}")
            field_weights = generate_weights_from_performance(performance_data, baseline_weight)

            # Create comprehensive report
            weights_report = generate_weights_report(field_weights, performance_data, unified_report_path)

            # Save weights report
            weights_path = os.path.join(self.output_dir, output_file)
            print(f"💾 Saving weights to: {os.path.basename(weights_path)}")
            with open(weights_path, 'w') as f:
                json.dump(weights_report, f, indent=2)

            # Print summary if verbose
            if verbose:
                print(f"\n📋 Generated Weights Summary:")
                for field_name, summary in weights_report["weight_summary"].items():
                    weights_str = ", ".join([f"{method}: {weight:.2f}"
                                           for method, weight in summary["weights"].items()])
                    print(f"   {field_name}: {weights_str}")
                    print(f"      → Dominant: {summary['dominant_method']} ({summary['dominant_weight']:.2f})")

                print(f"\n🔍 Performance Insights:")
                for field_name, insights in weights_report["performance_insights"].items():
                    print(f"   {field_name}:")
                    for insight in insights:
                        print(f"      • {insight}")

            print(f"✅ Detection weights generated successfully!")
            print(f"📁 Weights file: {os.path.basename(weights_path)}")

            return True

        except Exception as e:
            print(f"❌ Error generating weights: {e}")
            if verbose:
                traceback.print_exc()
            return False

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
  python single_sample_multi_field_demo.py                                    # Default 20% injection intensity
  python single_sample_multi_field_demo.py --injection-intensity 0.1         # 10% injection intensity
  python single_sample_multi_field_demo.py --injection-intensity 0.3 --max-issues-per-row 3  # Higher intensity
  python single_sample_multi_field_demo.py --data-file my_data.csv --output-dir my_results
  python single_sample_multi_field_demo.py --generate-weights --weights-verbose  # Generate detection weights
        """
    )
    parser.add_argument("--data-file",
                       help="Path to the CSV data file")
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

    # Brand configuration options
    parser.add_argument("--brand", help="Brand name (deprecated - uses static config)")
    parser.add_argument("--brand-config", help="Path to brand configuration JSON file (deprecated - uses static config)")
    parser.add_argument("--llm-context-columns", type=str, default=None,
                       help="Comma-separated list of context columns for LLM dynamic encoding")
    parser.add_argument("--use-weighted-combination", action="store_true",
                       help="Use weighted combination of anomaly detection methods instead of priority-based")
    parser.add_argument("--weights-file", type=str, default="detection_weights.json",
                       help="Path to JSON file containing detection weights (default: detection_weights.json)")

    # Weight generation options
    parser.add_argument("--generate-weights", action="store_true",
                       help="Generate detection weights after demo completion based on performance results")
    parser.add_argument("--weights-output-file", type=str, default="generated_detection_weights.json",
                       help="Output file for generated weights (default: generated_detection_weights.json)")
    parser.add_argument("--baseline-weight", type=float, default=0.1,
                       help="Baseline weight for untrained/poor performing methods (default: 0.1)")
    parser.add_argument("--weights-verbose", action="store_true",
                       help="Print detailed weight generation information")

    args = parser.parse_args()

    # Handle brand configuration
    if not args.brand:
        available_brands = get_available_brands()
        if len(available_brands) == 1:
            args.brand = available_brands[0]
            print(f"Using default brand: {args.brand}")
        elif len(available_brands) > 1:
            raise ConfigurationError(
                f"Multiple brands available. Please specify one with --brand\n"
                f"Available brands: {', '.join(available_brands)}"
            )
        else:
            raise ConfigurationError("No brand configurations found.")

    # Load brand configuration
    try:
        brand_config = load_brand_config(args.brand)
        print(f"Using brand configuration: {args.brand}")
    except FileNotFoundError as e:
        raise ConfigurationError(f"Brand configuration not found: {e}") from e

    # Use brand's data file if --data-file not provided
    if not args.data_file:
        if brand_config.default_data_path:
            args.data_file = brand_config.default_data_path
            print(f"Using brand data file: {args.data_file}")
        else:
            raise ConfigurationError(f"No data file configured for brand '{args.brand}'")

    # Validate arguments
    if not (0.0 <= args.injection_intensity <= 1.0):
        raise ValueError("injection-intensity must be between 0.0 and 1.0")

    if args.max_issues_per_row < 1:
        raise ValueError("max-issues-per-row must be at least 1")

    # Create and run demo
    demo = DataQualityDemo(
        data_file=args.data_file,
        brand_name=args.brand,
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
        llm_context_columns=args.llm_context_columns,
        use_weighted_combination=args.use_weighted_combination,
        weights_file=args.weights_file
    )

    result = demo.run_complete_demo()

    if result:
        print("\n✅ Demo completed successfully!")

        # Generate weights if requested
        if args.generate_weights:
            unified_report_path = result["report_files"]["unified_report"]
            weights_generated = demo.generate_detection_weights(
                unified_report_path=unified_report_path,
                output_file=args.weights_output_file,
                baseline_weight=args.baseline_weight,
                verbose=args.weights_verbose
            )

            if weights_generated:
                print(f"\n🔧 Detection weights generated and saved to: {args.weights_output_file}")
            else:
                print(f"\n⚠️  Weight generation failed - see error messages above")
    else:
        raise RuntimeError("Demo failed or was interrupted")


if __name__ == "__main__":
    main()
