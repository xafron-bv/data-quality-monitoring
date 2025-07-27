#!/usr/bin/env python3
"""
Test script to verify basic imports work for all modified files.
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Test if a module can be imported."""
    try:
        if '.' in module_name:
            # Handle submodule imports
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
        else:
            module = __import__(module_name)
        print(f"✅ {module_name:<50} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<50} - ImportError: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name:<50} - {type(e).__name__}: {e}")
        return False

def main():
    """Run import tests for all modified modules."""
    print("=" * 80)
    print("Testing Basic Imports for Modified Files")
    print("=" * 80)
    
    # Core utility modules
    print("\n1. Core Utility Modules:")
    test_import("field_mapper", "Field mapping interface")
    test_import("common_interfaces", "Common interfaces")
    test_import("exceptions", "Custom exceptions")
    test_import("debug_config", "Debug configuration")
    
    # Analysis and helper scripts
    print("\n2. Analysis and Helper Scripts:")
    test_import("analyze_column", "Column analysis utility")
    test_import("field_column_map", "Field to column mapping")
    test_import("brand_configs", "Brand configuration management")
    test_import("manage_brands", "Brand management CLI")
    
    # Error and anomaly injection
    print("\n3. Error and Anomaly Injection:")
    test_import("error_injection", "Error injection utilities")
    test_import("anomaly_detectors.anomaly_injection", "Anomaly injection")
    
    # Detection components
    print("\n4. Detection Components:")
    test_import("anomaly_detectors.ml_based.ml_anomaly_detector", "ML anomaly detector")
    test_import("anomaly_detectors.llm_based.llm_anomaly_detector", "LLM anomaly detector")
    test_import("anomaly_detectors.pattern_based.pattern_based_detector", "Pattern detector")
    
    # Training scripts
    print("\n5. Training Scripts:")
    test_import("anomaly_detectors.ml_based.model_training", "ML model training")
    test_import("anomaly_detectors.ml_based.hyperparameter_search", "Hyperparameter search")
    test_import("anomaly_detectors.ml_based.generate_centroids_for_existing_models", "Centroid generation")
    
    # Evaluation and demo
    print("\n6. Evaluation and Demo Scripts:")
    test_import("evaluator", "Evaluator base class")
    test_import("evaluate", "Main evaluation script")
    test_import("demo", "Demo script")
    test_import("comprehensive_detector", "Comprehensive detector")
    test_import("comprehensive_sample_generator", "Sample generator")
    
    # Reporting and analysis
    print("\n7. Reporting and Analysis:")
    test_import("ml_curve_generator", "ML curve generator")
    test_import("confusion_matrix_analyzer", "Confusion matrix analyzer")
    test_import("consolidated_reporter", "Consolidated reporter")
    test_import("detection_comparison", "Detection comparison")
    
    # Integration tests
    print("\n8. Integration Tests:")
    test_import("test_llm_integration", "LLM integration test")
    test_import("unified_detection_interface", "Unified detection interface")
    
    print("\n" + "=" * 80)
    print("Import test completed!")

if __name__ == "__main__":
    main()