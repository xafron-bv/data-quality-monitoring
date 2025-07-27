#!/usr/bin/env python3
"""
Weighted Anomaly Detection Combiner

This module combines results from multiple anomaly detection methods using 
field-specific weights calculated from performance metrics (precision, recall, F1-score).
Validation results are always applied first (highest priority), then anomaly 
detection results are combined using weighted probabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from comprehensive_detector import ComprehensiveFieldDetector, CellClassification, FieldDetectionResult


@dataclass
class DetectionWeight:
    """Weights for a detection method on a specific field."""
    field_name: str
    method: str  # "pattern_based", "ml_based", "llm_based"
    precision_weight: float
    recall_weight: float
    f1_weight: float
    combined_weight: float


class WeightedAnomalyDetector(ComprehensiveFieldDetector):
    """
    Enhanced detector that combines multiple anomaly detection methods using 
    field-specific weights based on performance metrics.
    """
    
    def __init__(self, field_mapper, enable_validation=True, enable_pattern=True, 
                 enable_ml=True, enable_llm=False, llm_threshold=0.6):
        super().__init__(field_mapper, enable_validation, enable_pattern, 
                        enable_ml, enable_llm, llm_threshold)
        
        # Field-specific weights calculated from performance analysis
        self.detection_weights = self._calculate_field_weights()
        
    def _calculate_field_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate field-specific weights for each detection method based on 
        performance analysis from demo results.
        
        Returns:
            Dict mapping field_name -> method -> weight
        """
        # Performance data from evaluation results
        performance_data = {
            "category": {
                "pattern_based": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "ml_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            },
            "color_name": {
                "pattern_based": {"precision": 0.968, "recall": 0.882, "f1": 0.923},
                "ml_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            },
            "material": {
                "pattern_based": {"precision": 0.045, "recall": 0.024, "f1": 0.032},
                "ml_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            },
            "size": {
                "pattern_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "ml_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            },
            "care_instructions": {
                "pattern_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "ml_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_based": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            }
        }
        
        # Calculate weights for each field
        field_weights = {}
        
        for field_name, methods in performance_data.items():
            field_weights[field_name] = {}
            
            # Calculate total F1 score across all methods for normalization
            total_f1 = sum(method_perf["f1"] for method_perf in methods.values())
            
            # If no method has performance, give equal weights
            if total_f1 == 0:
                num_methods = len(methods)
                for method in methods.keys():
                    field_weights[field_name][method] = 1.0 / num_methods
            else:
                # Weight methods by their F1 score performance
                for method, method_perf in methods.items():
                    # Use F1 score as primary weight, with small baseline for untrained methods
                    f1_score = method_perf["f1"]
                    baseline_weight = 0.1  # Small weight for untrained methods
                    field_weights[field_name][method] = max(f1_score, baseline_weight)
                
                # Normalize weights to sum to 1
                total_weight = sum(field_weights[field_name].values())
                for method in field_weights[field_name]:
                    field_weights[field_name][method] /= total_weight
        
        return field_weights
    
    def get_field_detection_weights(self, field_name: str) -> Dict[str, float]:
        """Get detection weights for a specific field."""
        return self.detection_weights.get(field_name, {
            "pattern_based": 0.33,
            "ml_based": 0.33, 
            "llm_based": 0.34
        })
    
    def weighted_classify_cells(self, df: pd.DataFrame, 
                               field_results: Dict[str, FieldDetectionResult]) -> List[CellClassification]:
        """
        Classify cells using weighted combination of detection methods.
        Validation results have highest priority, then anomaly detection results 
        are combined using field-specific weights.
        """
        print(f"   🎯 Classifying cells with WEIGHTED COMBINATION approach")
        
        # Dictionary to store weighted detection scores for each cell
        cell_scores = {}  # (row_idx, column_name) -> {method: score}
        validation_detections = {}  # (row_idx, column_name) -> validation_result
        
        # First pass: collect all detection results
        for field_name, result in field_results.items():
            column_name = result.column_name
            if not column_name:
                continue
            
            weights = self.get_field_detection_weights(field_name)
            
            # Store validation results (highest priority - always used)
            for detection in result.validation_results:
                row_idx = detection.get('row_index')
                if row_idx is not None:
                    key = (row_idx, column_name)
                    validation_detections[key] = {
                        'field_name': field_name,
                        'detection': detection,
                        'column_name': column_name
                    }
            
            # Collect anomaly detection scores for weighted combination
            methods_data = [
                ('pattern_based', result.anomaly_results),
                ('ml_based', result.ml_results),
                ('llm_based', result.llm_results)
            ]
            
            for method, detections in methods_data:
                method_weight = weights.get(method, 0.0)
                
                for detection in detections:
                    row_idx = detection.get('row_index')
                    if row_idx is not None:
                        key = (row_idx, column_name)
                        
                        # Skip if validation already found this cell
                        if key in validation_detections:
                            continue
                        
                        if key not in cell_scores:
                            cell_scores[key] = {
                                'field_name': field_name,
                                'column_name': column_name,
                                'methods': {},
                                'weighted_score': 0.0,
                                'best_detection': None
                            }
                        
                        # Store method-specific score
                        confidence = detection.get('probability', 0.5)
                        weighted_confidence = confidence * method_weight
                        
                        cell_scores[key]['methods'][method] = {
                            'confidence': confidence,
                            'weight': method_weight,
                            'weighted_confidence': weighted_confidence,
                            'detection': detection
                        }
                        
                        # Update total weighted score
                        cell_scores[key]['weighted_score'] += weighted_confidence
                        
                        # Track best individual detection for details
                        if (cell_scores[key]['best_detection'] is None or 
                            confidence > cell_scores[key]['best_detection'].get('probability', 0)):
                            cell_scores[key]['best_detection'] = detection
                            cell_scores[key]['best_method'] = method
        
        # Second pass: create cell classifications
        cell_classifications = []
        
        # Add validation results (highest priority)
        for key, val_data in validation_detections.items():
            row_idx, column_name = key
            detection = val_data['detection']
            
            try:
                original_value = df.at[row_idx, column_name]
            except (IndexError, KeyError):
                original_value = None
            
            classification = CellClassification(
                row_index=row_idx,
                column_name=column_name,
                field_name=val_data['field_name'],
                status='ERROR',
                message=detection.get('display_message', 'Validation error'),
                confidence=detection.get('probability', 1.0),
                detection_type='validation',
                original_value=original_value,
                detected_value=detection.get('error_data'),
                error_code=detection.get('error_code')
            )
            
            cell_classifications.append(classification)
        
        # Add weighted anomaly detection results
        anomaly_threshold = 0.3  # Minimum weighted score to classify as anomaly
        
        for key, score_data in cell_scores.items():
            if score_data['weighted_score'] >= anomaly_threshold:
                row_idx, column_name = key
                best_detection = score_data['best_detection']
                
                try:
                    original_value = df.at[row_idx, column_name]
                except (IndexError, KeyError):
                    original_value = None
                
                # Create detailed message showing contributing methods
                contributing_methods = []
                for method, method_data in score_data['methods'].items():
                    if method_data['weighted_confidence'] > 0:
                        contributing_methods.append(
                            f"{method}({method_data['confidence']:.2f}*{method_data['weight']:.2f})"
                        )
                
                message = f"Weighted anomaly (score: {score_data['weighted_score']:.3f}): {', '.join(contributing_methods)}"
                
                classification = CellClassification(
                    row_index=row_idx,
                    column_name=column_name,
                    field_name=score_data['field_name'],
                    status='ANOMALY',
                    message=message,
                    confidence=score_data['weighted_score'],
                    detection_type='weighted_anomaly',
                    original_value=original_value,
                    detected_value=best_detection.get('error_data') if best_detection else None,
                    error_code=best_detection.get('error_code') if best_detection else None
                )
                
                cell_classifications.append(classification)
        
        # Sort by row_index for easier navigation
        cell_classifications.sort(key=lambda x: (x.row_index, x.column_name))
        
        validation_count = len(validation_detections)
        anomaly_count = len([c for c in cell_classifications if c.detection_type == 'weighted_anomaly'])
        
        print(f"   ✅ Classified {len(cell_classifications)} cells with issues")
        print(f"       📝 Validation: {validation_count}")
        print(f"       🔍 Weighted anomalies: {anomaly_count}")
        
        return cell_classifications
    
    def run_weighted_detection(self, df: pd.DataFrame, 
                              selected_fields: Optional[List[str]] = None) -> Tuple[Dict[str, FieldDetectionResult], List[CellClassification]]:
        """
        Run detection using weighted combination approach.
        """
        print(f"🔍 Starting WEIGHTED detection on {len(df)} rows")
        
        # Run all detection methods first (reuse parent class logic)
        field_results, _ = super().run_comprehensive_detection(df, selected_fields)
        
        # Apply weighted classification
        weighted_classifications = self.weighted_classify_cells(df, field_results)
        
        # Print summary with weights information
        print(f"\n✅ Weighted detection complete:")
        print(f"   📊 Detection weights by field:")
        
        for field_name in field_results.keys():
            weights = self.get_field_detection_weights(field_name)
            weight_str = ", ".join([f"{method}: {weight:.2f}" for method, weight in weights.items()])
            print(f"       {field_name}: {weight_str}")
        
        total_issues = len(weighted_classifications)
        affected_rows = len(set(c.row_index for c in weighted_classifications))
        
        print(f"   📊 Total issues detected: {total_issues}")
        print(f"   🎯 Affected rows: {affected_rows} / {len(df)} ({affected_rows/len(df)*100:.1f}%)")
        
        return field_results, weighted_classifications
    
    def save_weights_report(self, output_file: str):
        """Save the detection weights to a JSON file for analysis."""
        weights_report = {
            "description": "Field-specific weights for anomaly detection methods",
            "calculation_method": "F1-score based with normalization",
            "weights": self.detection_weights,
            "performance_basis": {
                "category": "Pattern-based performs excellently (F1=1.0)",
                "color_name": "Pattern-based performs very well (F1=0.923)",
                "material": "Pattern-based performs poorly (F1=0.032), but still best available",
                "size": "No anomalies detected by any method",
                "care_instructions": "No anomalies detected by any method"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(weights_report, f, indent=2)
        
        print(f"💾 Saved detection weights report to: {output_file}")