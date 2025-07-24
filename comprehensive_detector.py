#!/usr/bin/env python3
"""
Comprehensive Field-by-Field Detector

This module runs detection methods (validation, pattern-based anomaly detection,
ML-based anomaly detection) on each field separately and classifies each cell
with priority: validation > pattern-based > ML-based.
"""

import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from common_interfaces import FieldMapper
from evaluator import Evaluator
from error_injection import load_error_rules
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector
from exceptions import ConfigurationError, FileOperationError


@dataclass
class CellClassification:
    """Classification result for a single cell."""
    row_index: int
    column_name: str
    field_name: str
    status: str  # "CLEAN", "ERROR", "ANOMALY"
    message: Optional[str] = None
    confidence: float = 0.0
    detection_type: Optional[str] = None  # "validation", "pattern_based", "ml_based"
    original_value: Any = None
    detected_value: Any = None


@dataclass
class FieldDetectionResult:
    """Detection results for a single field."""
    field_name: str
    column_name: str
    validation_results: List[Dict[str, Any]]
    anomaly_results: List[Dict[str, Any]]
    ml_results: List[Dict[str, Any]]
    total_issues: int
    detection_summary: Dict[str, int]


def load_module_class(module_path: str):
    """Dynamically loads a class from a Python file based on a module path string."""
    import importlib
    try:
        module_name, class_name = module_path.split(':')
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ConfigurationError(f"Could not load class from '{module_path}': {e}")


class ComprehensiveFieldDetector:
    """
    Runs all available detection methods on each field separately and classifies cells.
    """
    
    def __init__(self, field_mapper: FieldMapper, 
                 validation_threshold: float = 0.0,
                 anomaly_threshold: float = 0.7,
                 ml_threshold: float = 0.7,
                 batch_size: int = 1000,
                 max_workers: int = 2,
                 core_fields_only: bool = False):
        """
        Initialize comprehensive field detector.
        
        Args:
            field_mapper: FieldMapper instance for field-column mapping
            validation_threshold: Minimum confidence for validation results
            anomaly_threshold: Minimum confidence for anomaly detection results
            ml_threshold: Minimum confidence for ML detection results
            batch_size: Batch size for detection operations
            max_workers: Maximum number of worker threads
            core_fields_only: If True, only analyze core fields to save memory
        """
        self.field_mapper = field_mapper
        self.validation_threshold = validation_threshold
        self.anomaly_threshold = anomaly_threshold
        self.ml_threshold = ml_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.core_fields_only = core_fields_only
        
        # Core fields that have good validation/anomaly detection
        self.core_fields = {"material", "color_name", "category", "size", "care_instructions"}
        
        # Cache for loaded components
        self._validator_cache = {}
        self._anomaly_detector_cache = {}
        self._ml_detector_cache = {}
    
    def get_available_detection_fields(self) -> Dict[str, Dict[str, bool]]:
        """
        Get fields that have detection capabilities available.
        
        Returns:
            Dict mapping field names to their available detection methods:
            {
                "material": {"validation": True, "anomaly": True, "ml": True},
                "color_name": {"validation": True, "anomaly": True, "ml": False},
                ...
            }
        """
        available_fields = {}
        
        for field_name in self.field_mapper.get_available_fields():
            # Skip non-core fields if core_fields_only is enabled
            if self.core_fields_only and field_name not in self.core_fields:
                continue
                
            capabilities = {
                "validation": self._has_validation_capability(field_name),
                "anomaly": self._has_anomaly_capability(field_name),
                "ml": self._has_ml_capability(field_name)
            }
            
            # Only include fields that have at least one detection method
            if any(capabilities.values()):
                available_fields[field_name] = capabilities
        
        return available_fields
    
    def _has_validation_capability(self, field_name: str) -> bool:
        """Check if validation is available for a field."""
        try:
            validator_module_str = f"validators.{field_name}.validate:Validator"
            load_module_class(validator_module_str)
            return True
        except:
            return False
    
    def _has_anomaly_capability(self, field_name: str) -> bool:
        """Check if pattern-based anomaly detection is available for a field."""
        try:
            anomaly_detector_module_str = f"anomaly_detectors.pattern_based.{field_name}.detect:AnomalyDetector"
            load_module_class(anomaly_detector_module_str)
            return True
        except:
            return False
    
    def _has_ml_capability(self, field_name: str) -> bool:
        """Check if ML-based anomaly detection is available for a field."""
        try:
            ml_detector = MLAnomalyDetector(field_name=field_name, threshold=self.ml_threshold)
            return True
        except:
            return False
    
    def _get_validator_components(self, field_name: str) -> Tuple[Any, Any]:
        """Get or create validator and reporter for a field."""
        if field_name not in self._validator_cache:
            try:
                validator_module_str = f"validators.{field_name}.validate:Validator"
                reporter_module_str = f"validators.report:Reporter"
                
                ValidatorClass = load_module_class(validator_module_str)
                ReporterClass = load_module_class(reporter_module_str)
                
                validator = ValidatorClass()
                reporter = ReporterClass(field_name)
                
                self._validator_cache[field_name] = (validator, reporter)
            except Exception as e:
                print(f"      ⚠️  Could not load validation for {field_name}: {e}")
                self._validator_cache[field_name] = (None, None)
        
        return self._validator_cache[field_name]
    
    def _get_anomaly_components(self, field_name: str) -> Tuple[Any, Any]:
        """Get or create anomaly detector and reporter for a field."""
        if field_name not in self._anomaly_detector_cache:
            try:
                detector_module_str = f"anomaly_detectors.pattern_based.{field_name}.detect:AnomalyDetector"
                reporter_module_str = f"anomaly_detectors.pattern_based.report:AnomalyReporter"
                
                DetectorClass = load_module_class(detector_module_str)
                ReporterClass = load_module_class(reporter_module_str)
                
                detector = DetectorClass()
                reporter = ReporterClass(field_name)
                
                self._anomaly_detector_cache[field_name] = (detector, reporter)
            except Exception as e:
                print(f"      ⚠️  Could not load anomaly detection for {field_name}: {e}")
                self._anomaly_detector_cache[field_name] = (None, None)
        
        return self._anomaly_detector_cache[field_name]
    
    def _get_ml_detector(self, field_name: str) -> Any:
        """Get or create ML detector for a field."""
        # Don't cache ML detectors to save memory - create fresh each time
        try:
            ml_detector = MLAnomalyDetector(field_name=field_name, threshold=self.ml_threshold)
            return ml_detector
        except Exception as e:
            print(f"      ⚠️  Could not load ML detection for {field_name}: {e}")
            return None
    
    def detect_field_issues(self, df: pd.DataFrame, field_name: str) -> FieldDetectionResult:
        """
        Run all available detection methods on a single field.
        
        Args:
            df: DataFrame to analyze
            field_name: Field to analyze
            
        Returns:
            FieldDetectionResult with all detection results
        """
        try:
            column_name = self.field_mapper.validate_column_exists(df, field_name)
        except ValueError:
            print(f"      ❌ Field {field_name} not found in DataFrame")
            return FieldDetectionResult(
                field_name=field_name,
                column_name="",
                validation_results=[],
                anomaly_results=[],
                ml_results=[],
                total_issues=0,
                detection_summary={}
            )
        
        print(f"      🔍 Analyzing field '{field_name}' (column: '{column_name}')")
        
        validation_results = []
        anomaly_results = []
        ml_results = []
        
        # Run validation if available
        validator, validator_reporter = self._get_validator_components(field_name)
        if validator and validator_reporter:
            try:
                validation_errors = validator.bulk_validate(df, column_name)
                validation_report = validator_reporter.generate_report(validation_errors, df)
                validation_results = [r for r in validation_report if r.get('probability', 1.0) >= self.validation_threshold]
                print(f"         📝 Validation: {len(validation_results)} errors found")
            except Exception as e:
                print(f"         ⚠️  Validation failed: {e}")
        
        # Run pattern-based anomaly detection if available
        anomaly_detector, anomaly_reporter = self._get_anomaly_components(field_name)
        if anomaly_detector and anomaly_reporter:
            try:
                # Pattern detectors should work with fixed rules, no learning during detection
                anomalies = anomaly_detector.bulk_detect(df, column_name, self.batch_size, max_workers=1)
                anomaly_report = anomaly_reporter.generate_report(anomalies, df)
                anomaly_results = [r for r in anomaly_report if r.get('probability', 0.7) >= self.anomaly_threshold]
                print(f"         🔍 Pattern anomalies: {len(anomaly_results)} anomalies found")
            except Exception as e:
                print(f"         ⚠️  Pattern anomaly detection failed: {e}")
        
        # Run ML-based anomaly detection if available
        ml_detector = self._get_ml_detector(field_name)
        if ml_detector:
            try:
                # Use single-threaded processing to save memory
                ml_anomalies = ml_detector.bulk_detect(df, column_name, self.batch_size, max_workers=1)
                # ML detector returns results in a different format, normalize them
                ml_results_formatted = []
                for anomaly in ml_anomalies:
                    if anomaly.probability >= self.ml_threshold:
                        ml_results_formatted.append({
                            'row_index': anomaly.row_index,
                            'column_name': column_name,
                            'error_data': anomaly.anomaly_data,
                            'display_message': f"ML Anomaly: {getattr(anomaly, 'explanation', 'Semantic similarity anomaly detected')}",
                            'probability': anomaly.probability,
                            'detection_type': 'ml_based'
                        })
                ml_results = ml_results_formatted
                print(f"         🤖 ML anomalies: {len(ml_results)} anomalies found")
            except Exception as e:
                print(f"         ⚠️  ML anomaly detection failed: {e}")
            
            # Force cleanup of ML detector immediately after use
            del ml_detector
            import gc
            gc.collect()
        
        total_issues = len(validation_results) + len(anomaly_results) + len(ml_results)
        detection_summary = {
            "validation_errors": len(validation_results),
            "pattern_anomalies": len(anomaly_results),
            "ml_anomalies": len(ml_results)
        }
        
        return FieldDetectionResult(
            field_name=field_name,
            column_name=column_name,
            validation_results=validation_results,
            anomaly_results=anomaly_results,
            ml_results=ml_results,
            total_issues=total_issues,
            detection_summary=detection_summary
        )
    
    def classify_cells(self, df: pd.DataFrame, field_results: Dict[str, FieldDetectionResult]) -> List[CellClassification]:
        """
        Classify each cell based on detection results with priority: validation > pattern-based > ML-based.
        
        Args:
            df: Original DataFrame
            field_results: Detection results for all fields
            
        Returns:
            List of cell classifications (excludes clean cells)
        """
        print(f"   🎯 Classifying cells with priority: validation > pattern-based > ML-based")
        
        cell_classifications = []
        
        # Create a map of (row_index, column_name) -> detection results for quick lookup
        detection_map = {}
        
        for field_name, result in field_results.items():
            column_name = result.column_name
            if not column_name:  # Skip if field not found in DataFrame
                continue
            
            # Add validation results (highest priority)
            for detection in result.validation_results:
                row_idx = detection.get('row_index')
                if row_idx is not None:
                    key = (row_idx, column_name)
                    if key not in detection_map:
                        detection_map[key] = {
                            'field_name': field_name,
                            'priority': 1,  # Highest priority
                            'status': 'ERROR',
                            'detection_type': 'validation',
                            'message': detection.get('display_message', 'Validation error'),
                            'confidence': detection.get('probability', 1.0),
                            'detected_value': detection.get('error_data')
                        }
            
            # Add pattern-based anomaly results (medium priority)
            for detection in result.anomaly_results:
                row_idx = detection.get('row_index')
                if row_idx is not None:
                    key = (row_idx, column_name)
                    if key not in detection_map or detection_map[key]['priority'] > 2:
                        detection_map[key] = {
                            'field_name': field_name,
                            'priority': 2,  # Medium priority
                            'status': 'ANOMALY',
                            'detection_type': 'pattern_based',
                            'message': detection.get('display_message', 'Pattern-based anomaly'),
                            'confidence': detection.get('probability', 0.7),
                            'detected_value': detection.get('error_data')
                        }
            
            # Add ML-based anomaly results (lowest priority)
            for detection in result.ml_results:
                row_idx = detection.get('row_index')
                if row_idx is not None:
                    key = (row_idx, column_name)
                    if key not in detection_map or detection_map[key]['priority'] > 3:
                        detection_map[key] = {
                            'field_name': field_name,
                            'priority': 3,  # Lowest priority
                            'status': 'ANOMALY',
                            'detection_type': 'ml_based',
                            'message': detection.get('display_message', 'ML-based anomaly'),
                            'confidence': detection.get('probability', 0.5),
                            'detected_value': detection.get('error_data')
                        }
        
        # Convert detection map to cell classifications
        for (row_idx, column_name), detection_info in detection_map.items():
            try:
                original_value = df.at[row_idx, column_name]
            except (IndexError, KeyError):
                original_value = None
            
            classification = CellClassification(
                row_index=row_idx,
                column_name=column_name,
                field_name=detection_info['field_name'],
                status=detection_info['status'],
                message=detection_info['message'],
                confidence=detection_info['confidence'],
                detection_type=detection_info['detection_type'],
                original_value=original_value,
                detected_value=detection_info['detected_value']
            )
            
            cell_classifications.append(classification)
        
        # Sort by row_index for easier navigation
        cell_classifications.sort(key=lambda x: (x.row_index, x.column_name))
        
        print(f"   ✅ Classified {len(cell_classifications)} cells with issues")
        return cell_classifications
    
    def run_comprehensive_detection(self, df: pd.DataFrame, 
                                  selected_fields: Optional[List[str]] = None) -> Tuple[Dict[str, FieldDetectionResult], List[CellClassification]]:
        """
        Run comprehensive detection across all available fields SEQUENTIALLY to save memory.
        
        Args:
            df: DataFrame to analyze
            selected_fields: Optional list of specific fields to analyze (analyzes all if None)
            
        Returns:
            Tuple of (field_results, cell_classifications)
        """
        print(f"🔍 Starting comprehensive detection on {len(df)} rows (SEQUENTIAL MODE)")
        
        # Get available detection fields
        available_fields = self.get_available_detection_fields()
        
        # Filter to selected fields if specified
        if selected_fields:
            available_fields = {
                field: capabilities for field, capabilities in available_fields.items()
                if field in selected_fields
            }
        
        if not available_fields:
            print("❌ No fields with detection capabilities found")
            return {}, []
        
        print(f"📋 Will analyze {len(available_fields)} fields SEQUENTIALLY:")
        for field_name, capabilities in available_fields.items():
            methods = [method for method, available in capabilities.items() if available]
            print(f"   {field_name}: {', '.join(methods)}")
        
        # Process fields ONE AT A TIME to save memory
        field_results = {}
        all_cell_classifications = []
        
        for i, field_name in enumerate(available_fields.keys(), 1):
            print(f"\n   🔄 Processing field {i}/{len(available_fields)}: {field_name}")
            
            # Clear all cached detectors before processing next field to save memory
            if hasattr(self, '_validator_cache'):
                self._validator_cache.clear()
            if hasattr(self, '_anomaly_detector_cache'):
                self._anomaly_detector_cache.clear()
            if hasattr(self, '_ml_detector_cache'):
                self._ml_detector_cache.clear()
            
            # Process this single field
            result = self.detect_field_issues(df, field_name)
            field_results[field_name] = result
            
            # Convert this field's results to cell classifications immediately
            temp_field_results = {field_name: result}
            field_classifications = self.classify_cells(df, temp_field_results)
            all_cell_classifications.extend(field_classifications)
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            print(f"      ✅ {field_name}: {result.total_issues} issues found, {len(field_classifications)} cells classified")
        
        # Print final summary
        total_issues = sum(len(result.validation_results) + len(result.anomaly_results) + len(result.ml_results) 
                          for result in field_results.values())
        affected_rows = len(set(classification.row_index for classification in all_cell_classifications))
        
        print(f"\n✅ Comprehensive detection complete:")
        print(f"   📊 Total issues detected: {total_issues}")
        print(f"   🎯 Affected rows: {affected_rows} / {len(df)} ({affected_rows/len(df)*100:.1f}%)")
        print(f"   📋 Issues by type:")
        
        by_type = {}
        for classification in all_cell_classifications:
            detection_type = classification.detection_type
            by_type[detection_type] = by_type.get(detection_type, 0) + 1
        
        for detection_type, count in by_type.items():
            print(f"      {detection_type}: {count}")
        
        return field_results, all_cell_classifications 