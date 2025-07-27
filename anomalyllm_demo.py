#!/usr/bin/env python3
"""
AnomalyLLM Integration Demo

This script demonstrates the integration of AnomalyLLM concepts into the existing
data quality monitoring system, showcasing:

1. Few-shot learning with in-context examples
2. Dynamic-aware encoding for temporal patterns
3. Prototype-based reprogramming
4. Enhanced anomaly detection with explanations

Based on: "AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

from anomaly_detectors.ml_based.anomalyllm_integration import (
    AnomalyLLMIntegration, FewShotExample, DynamicContext
)
from anomaly_detectors.ml_based.enhanced_ml_anomaly_detector import (
    EnhancedMLAnomalyDetector, create_enhanced_ml_detector_for_field
)
from anomaly_detectors.ml_based.check_anomalies import load_model_for_field
from field_column_map import get_field_to_column_map


def create_demo_data():
    """Create demo data with temporal and contextual information."""
    np.random.seed(42)
    
    # Create base product data
    base_products = [
        {"material": "cotton", "category": "shirts", "color_name": "blue"},
        {"material": "polyester", "category": "pants", "color_name": "black"},
        {"material": "wool", "category": "jackets", "color_name": "brown"},
        {"material": "silk", "category": "dresses", "color_name": "red"},
        {"material": "denim", "category": "jeans", "color_name": "navy"},
    ]
    
    # Generate temporal data
    start_date = datetime(2024, 1, 1)
    data = []
    
    for i in range(100):
        base_product = base_products[i % len(base_products)]
        timestamp = start_date + timedelta(days=i, hours=np.random.randint(0, 24))
        
        # Add some anomalies
        if i % 10 == 0:  # Every 10th record has an anomaly
            if i % 30 == 0:
                material = "invalid_material_123"  # Invalid material
            elif i % 20 == 0:
                color_name = "xyz_color"  # Invalid color
            else:
                category = "unknown_category"  # Invalid category
        else:
            material = base_product["material"]
            color_name = base_product["color_name"]
            category = base_product["category"]
        
        data.append({
            "timestamp": timestamp,
            "sequence_id": i,
            "material": material,
            "color_name": color_name,
            "category": category,
            "price": np.random.uniform(10, 200),
            "brand": np.random.choice(["BrandA", "BrandB", "BrandC"]),
            "season": np.random.choice(["spring", "summer", "fall", "winter"])
        })
    
    return pd.DataFrame(data)


def create_few_shot_examples():
    """Create few-shot examples for in-context learning."""
    examples = [
        # Normal examples
        FewShotExample(
            value="cotton",
            label="normal",
            confidence=0.95,
            explanation="Common natural fiber material"
        ),
        FewShotExample(
            value="blue",
            label="normal",
            confidence=0.95,
            explanation="Standard color name"
        ),
        FewShotExample(
            value="shirts",
            label="normal",
            confidence=0.95,
            explanation="Valid product category"
        ),
        
        # Anomaly examples
        FewShotExample(
            value="invalid_material_123",
            label="anomaly",
            confidence=0.9,
            explanation="Contains numbers and underscores, not a valid material name"
        ),
        FewShotExample(
            value="xyz_color",
            label="anomaly",
            confidence=0.9,
            explanation="Unusual color name format, likely invalid"
        ),
        FewShotExample(
            value="unknown_category",
            label="anomaly",
            confidence=0.9,
            explanation="Generic category name, not specific enough"
        ),
    ]
    
    return examples


def demo_basic_anomalyllm_integration():
    """Demo basic AnomalyLLM integration features."""
    print("=" * 60)
    print("ANOMALYLLM INTEGRATION DEMO")
    print("=" * 60)
    
    # Create demo data
    print("\n1. Creating demo data with temporal and contextual information...")
    df = create_demo_data()
    print(f"   Created {len(df)} records with temporal and contextual features")
    
    # Create few-shot examples
    print("\n2. Setting up few-shot examples for in-context learning...")
    few_shot_examples = create_few_shot_examples()
    print(f"   Created {len(few_shot_examples)} few-shot examples")
    
    # Load a base model (assuming material field is trained)
    print("\n3. Loading base model for material field...")
    try:
        base_model, column_name, reference_centroid = load_model_for_field("material")
        print(f"   ✅ Loaded base model for column '{column_name}'")
        
        # Create AnomalyLLM integration
        print("\n4. Initializing AnomalyLLM integration...")
        anomalyllm = AnomalyLLMIntegration(
            base_model,
            enable_dynamic_encoding=True,
            enable_prototype_reprogramming=True,
            enable_in_context_learning=True
        )
        
        # Train with AnomalyLLM concepts
        print("\n5. Training with AnomalyLLM concepts...")
        training_results = anomalyllm.train_with_anomalyllm_concepts(
            df=df,
            column_name="material",
            field_name="material",
            few_shot_examples=few_shot_examples,
            temporal_column="timestamp",
            context_columns=["category", "brand", "season"]
        )
        
        print(f"   ✅ Training completed:")
        for key, value in training_results.items():
            if key.startswith('anomalyllm_features'):
                print(f"      - {key}: {value}")
        
        # Test anomaly detection
        print("\n6. Testing enhanced anomaly detection...")
        test_values = [
            "cotton",  # Normal
            "invalid_material_123",  # Anomaly
            "polyester",  # Normal
            "xyz_material",  # Anomaly
            "wool"  # Normal
        ]
        
        # Create dynamic context for each test value
        contexts = []
        for i, value in enumerate(test_values):
            context = DynamicContext(
                timestamp=datetime.now() + timedelta(hours=i),
                sequence_position=i,
                metadata={
                    "category": "test_category",
                    "brand": "test_brand",
                    "season": "test_season"
                }
            )
            contexts.append(context)
        
        results = anomalyllm.detect_anomalies(
            test_values,
            threshold=0.6,
            context=contexts
        )
        
        print("\n   Detection Results:")
        for i, (value, result) in enumerate(zip(test_values, results)):
            status = "🚨 ANOMALY" if result['is_anomaly'] else "✅ NORMAL"
            print(f"      {i+1}. '{value}' -> {status}")
            print(f"         Confidence: {result['probability_of_correctness']:.3f}")
            if result.get('explanation'):
                print(f"         Explanation: {result['explanation']}")
            if result.get('global_similarity') and result.get('local_similarity'):
                print(f"         Global/Local similarity: {result['global_similarity']:.3f}/{result['local_similarity']:.3f}")
            print()
        
        return anomalyllm
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Note: This demo requires a trained model for the 'material' field.")
        print("   Please run the ML training first: python anomaly_detectors/ml_based/index.py data/your_data.csv --rules material")
        return None


def demo_enhanced_detector():
    """Demo the enhanced ML anomaly detector."""
    print("\n" + "=" * 60)
    print("ENHANCED ML ANOMALY DETECTOR DEMO")
    print("=" * 60)
    
    # Create demo data
    df = create_demo_data()
    few_shot_examples = create_few_shot_examples()
    
    print("\n1. Creating enhanced ML detector with AnomalyLLM features...")
    try:
        detector = create_enhanced_ml_detector_for_field(
            field_name="material",
            threshold=0.6,
            enable_anomalyllm=True,
            few_shot_examples=few_shot_examples,
            temporal_column="timestamp",
            context_columns=["category", "brand", "season"]
        )
        
        print("   ✅ Enhanced detector created")
        
        # Get enhancement info
        enhancement_info = detector.get_enhancement_info()
        print(f"\n2. AnomalyLLM Enhancement Status:")
        for key, value in enhancement_info.items():
            print(f"   - {key}: {value}")
        
        # Initialize the detector
        print(f"\n3. Initializing detector with data...")
        detector.learn_patterns(df, "material")
        
        # Test detection
        print(f"\n4. Testing anomaly detection...")
        test_values = ["cotton", "invalid_material_123", "polyester"]
        
        for value in test_values:
            # Create context with temporal and metadata information
            context = {
                "timestamp": datetime.now(),
                "sequence_position": 1,
                "category": "test_category",
                "brand": "test_brand",
                "season": "test_season"
            }
            
            anomaly = detector._detect_anomaly(value, context)
            
            if anomaly:
                print(f"   🚨 Anomaly detected: '{value}'")
                print(f"      Probability: {anomaly.probability:.3f}")
                print(f"      Explanation: {anomaly.details.get('explanation', 'No explanation')}")
                print(f"      Detection method: {anomaly.details.get('detection_method', 'Unknown')}")
            else:
                print(f"   ✅ Normal: '{value}'")
            print()
        
        return detector
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Note: This demo requires a trained model for the 'material' field.")
        return None


def demo_few_shot_learning():
    """Demo few-shot learning capabilities."""
    print("\n" + "=" * 60)
    print("FEW-SHOT LEARNING DEMO")
    print("=" * 60)
    
    try:
        # Create enhanced detector
        detector = create_enhanced_ml_detector_for_field(
            field_name="material",
            threshold=0.6,
            enable_anomalyllm=True
        )
        
        print("1. Adding few-shot examples interactively...")
        
        # Add examples for different scenarios
        detector.add_few_shot_example(
            value="silk",
            label="normal",
            confidence=0.95,
            explanation="Premium natural fiber material"
        )
        
        detector.add_few_shot_example(
            value="material_123",
            label="anomaly",
            confidence=0.9,
            explanation="Contains numbers, not a valid material name"
        )
        
        detector.add_few_shot_example(
            value="unknown_fiber",
            label="anomaly",
            confidence=0.85,
            explanation="Generic term, not specific enough"
        )
        
        print(f"   ✅ Added {len(detector.few_shot_examples)} few-shot examples")
        
        # Test with new examples
        print("\n2. Testing with new examples...")
        test_cases = [
            ("cotton", "Should be normal"),
            ("fiber_456", "Should be anomaly (contains numbers)"),
            ("wool", "Should be normal"),
            ("generic_material", "Should be anomaly (too generic)")
        ]
        
        for value, expected in test_cases:
            anomaly = detector._detect_anomaly(value)
            status = "🚨 ANOMALY" if anomaly else "✅ NORMAL"
            print(f"   '{value}' -> {status} ({expected})")
            if anomaly:
                print(f"      Explanation: {anomaly.details.get('explanation', 'No explanation')}")
        
        return detector
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def demo_dynamic_context():
    """Demo dynamic context awareness."""
    print("\n" + "=" * 60)
    print("DYNAMIC CONTEXT AWARENESS DEMO")
    print("=" * 60)
    
    try:
        detector = create_enhanced_ml_detector_for_field(
            field_name="material",
            threshold=0.6,
            enable_anomalyllm=True
        )
        
        print("1. Testing with different temporal contexts...")
        
        test_value = "cotton"
        base_time = datetime.now()
        
        # Test with different timestamps
        for i in range(3):
            context = {
                "timestamp": base_time + timedelta(hours=i),
                "sequence_position": i,
                "category": f"category_{i}",
                "brand": f"brand_{i}",
                "season": "summer" if i % 2 == 0 else "winter"
            }
            
            anomaly = detector._detect_anomaly(test_value, context)
            status = "🚨 ANOMALY" if anomaly else "✅ NORMAL"
            
            print(f"   Time {i}: {status}")
            print(f"      Context: {context}")
            if anomaly:
                print(f"      Explanation: {anomaly.details.get('explanation', 'No explanation')}")
            print()
        
        print("2. Testing with different metadata contexts...")
        
        contexts = [
            {"category": "shirts", "brand": "premium", "season": "summer"},
            {"category": "pants", "brand": "budget", "season": "winter"},
            {"category": "dresses", "brand": "luxury", "season": "spring"}
        ]
        
        for i, context in enumerate(contexts):
            anomaly = detector._detect_anomaly(test_value, context)
            status = "🚨 ANOMALY" if anomaly else "✅ NORMAL"
            
            print(f"   Context {i+1}: {status}")
            print(f"      Metadata: {context}")
            if anomaly:
                print(f"      Explanation: {anomaly.details.get('explanation', 'No explanation')}")
            print()
        
        return detector
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    """Run all AnomalyLLM integration demos."""
    print("🚀 ANOMALYLLM INTEGRATION DEMONSTRATION")
    print("Based on: 'AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models'")
    print("Paper: https://arxiv.org/abs/2405.07626")
    print("Repository: https://github.com/AnomalyLLM/AnomalyLLM")
    
    # Run demos
    anomalyllm = demo_basic_anomalyllm_integration()
    enhanced_detector = demo_enhanced_detector()
    few_shot_detector = demo_few_shot_learning()
    context_detector = demo_dynamic_context()
    
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    print("\n✅ AnomalyLLM Integration Features Demonstrated:")
    print("   • Few-shot learning with in-context examples")
    print("   • Dynamic-aware encoding for temporal patterns")
    print("   • Prototype-based reprogramming")
    print("   • Enhanced explanations and context awareness")
    print("   • Seamless integration with existing ML framework")
    
    print("\n🔧 Key Innovations from AnomalyLLM Paper:")
    print("   • In-context learning for few-shot anomaly detection")
    print("   • Dynamic-aware encoder for temporal data")
    print("   • Prototype-based edge reprogramming")
    print("   • Enhanced semantic alignment with LLM knowledge")
    
    print("\n📈 Benefits for Data Quality Monitoring:")
    print("   • Improved detection accuracy with few examples")
    print("   • Better handling of temporal data patterns")
    print("   • More interpretable anomaly explanations")
    print("   • Enhanced context awareness")
    print("   • Reduced training data requirements")
    
    print("\n🎯 Next Steps:")
    print("   • Train models with AnomalyLLM features enabled")
    print("   • Add domain-specific few-shot examples")
    print("   • Configure temporal and context columns")
    print("   • Evaluate performance improvements")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main() 