"""
Comprehensive field detector that orchestrates multiple detection methods.
"""

import os
import sys
import json
import pandas as pd
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import importlib

from field_mapper import FieldMapper
from common_interfaces import AnomalyIssue
from exceptions import ConfigurationError, DataError, FileOperationError, ModelError
from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.pattern_based.pattern_based_detector import PatternBasedDetector
from anomaly_detectors.pattern_based.report import AnomalyReporter
from debug_config import debug_print
from validators.validator_interface import ValidatorInterface
from anomaly_detectors.ml_based.ml_anomaly_detector import MLAnomalyDetector

# Optional imports with graceful fallback
try:
    from anomaly_detectors.llm_based.llm_anomaly_detector import create_llm_detector_for_field
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


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
    error_code: Optional[str] = None


@dataclass
class FieldDetectionResult:
    """Detection results for a single field."""
    field_name: str
    column_name: str
    validation_results: List[Dict[str, Any]]
    anomaly_results: List[Dict[str, Any]]
    ml_results: List[Dict[str, Any]]
    llm_results: List[Dict[str, Any]]
    total_issues: int
    detection_summary: Dict[str, int]


def load_module_class(module_path: str):
    """Dynamically loads a class from a Python file based on a module path string."""
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
                 llm_threshold: float = 0.6,
                 batch_size: int = 1000,
                 max_workers: int = 2,
                 core_fields_only: bool = False,
                 enable_validation: bool = True,
                 enable_pattern: bool = True,
                 enable_ml: bool = True,
                 enable_llm: bool = False,
                 use_weighted_combination: bool = False,
                 weights_file: str = "detection_weights.json"):
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
            enable_validation: If True, run validation detection
            enable_pattern: If True, run pattern-based anomaly detection
            enable_ml: If True, run ML-based anomaly detection
            enable_llm: If True, run LLM-based anomaly detection
            use_weighted_combination: If True, use weighted combination instead of priority-based
            weights_file: Path to JSON file containing detection weights
        """
        self.field_mapper = field_mapper
        self.validation_threshold = validation_threshold
        self.anomaly_threshold = anomaly_threshold
        self.ml_threshold = ml_threshold
        self.llm_threshold = llm_threshold
        self.batch_size = batch_size
        self.use_weighted_combination = use_weighted_combination
        self.weights_file = weights_file
        self.max_workers = max_workers
        self.core_fields_only = core_fields_only
        self.enable_validation = enable_validation
        self.enable_pattern = enable_pattern
        self.enable_ml = enable_ml
        self.enable_llm = enable_llm
        
        # Core fields that have good validation/anomaly detection
        self.core_fields = {"material", "color_name", "category", "size", "care_instructions"}
        
        # Cache for loaded components
        self._validator_cache = {}
        self._anomaly_detector_cache = {}
        self._ml_detector_cache = {}
        
        # Initialize detection weights if using weighted combination
        if self.use_weighted_combination:
            self.detection_weights = self._load_field_weights_from_file(self.weights_file)
    
    def _load_field_weights_from_file(self, weights_file: str = "detection_weights.json") -> Dict[str, Dict[str, float]]:
        """
        Load field-specific weights from a JSON file generated by generate_detection_weights.py
        
        Args:
            weights_file: Path to the weights JSON file
            
        Returns:
            Dict mapping field_name -> method -> weight
        """
        try:
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
            
            field_weights = weights_data.get('weights', {})
            print(f"✅ Loaded detection weights from: {weights_file}")
            
            # Validate weights structure
            for field_name, weights in field_weights.items():
                if not isinstance(weights, dict):
                    raise ValueError(f"Invalid weights format for field '{field_name}'")
                
                # Ensure all required methods have weights
                required_methods = ["pattern_based", "ml_based", "llm_based"]
                for method in required_methods:
                    if method not in weights:
                        weights[method] = 0.33  # Default equal weight
            
            return field_weights
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Warning: Could not load weights from {weights_file}: {e}")
            print(f"⚠️  Falling back to equal weights for all methods")
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """Get default equal weights for all methods when weights file is not available."""
        default_weights = {
            "pattern_based": 0.33,
            "ml_based": 0.33,
            "llm_based": 0.34
        }
        
        # Apply default weights to common fields
        common_fields = ["category", "color_name", "material", "size", "care_instructions"]
        return {field: default_weights.copy() for field in common_fields}
    
    def get_field_detection_weights(self, field_name: str) -> Dict[str, float]:
        """Get detection weights for a specific field."""
        if not self.use_weighted_combination:
            return {"pattern_based": 0.33, "ml_based": 0.33, "llm_based": 0.34}
        
        return self.detection_weights.get(field_name, {
            "pattern_based": 0.33,
            "ml_based": 0.33, 
            "llm_based": 0.34
        })

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
                "ml": self._has_ml_capability(field_name),
                "llm": self._has_llm_capability(field_name)
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
    
    def _has_llm_capability(self, field_name: str) -> bool:
        """Check if LLM-based anomaly detection is available for a field."""
        try:
            llm_detector = LLMAnomalyDetector(field_name=field_name, threshold=self.ml_threshold)
            return True
        except:
            return False
    
    def _get_validator_components(self, field_name: str) -> Tuple[Any, Any]:
        """Get or create validator and reporter for a field.
        
        Note: Validators are global and work across all brands. The field_name
        refers to the standard field name (e.g., 'material'), not the brand-specific
        column name. The FieldMapper handles the column name translation.
        """
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
        """Get or create anomaly detector and reporter for a field.
        
        Note: Pattern-based anomaly detectors are global and work across all brands.
        The field_name refers to the standard field name (e.g., 'material'), not the
        brand-specific column name. The FieldMapper handles the column name translation.
        """
        if field_name not in self._anomaly_detector_cache:
            try:
                # Use the new generic pattern-based detector
                detector = PatternBasedDetector(field_name)
                reporter = AnomalyReporter(field_name)
                
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
    
    def _get_llm_detector(self, field_name: str) -> Any:
        """Get or create LLM detector for a field."""
        # Don't cache LLM detectors to save memory - create fresh each time
        try:
            llm_detector = LLMAnomalyDetector(field_name=field_name, threshold=self.llm_threshold)
            return llm_detector
        except Exception as e:
            print(f"      ⚠️  Could not load LLM detection for {field_name}: {e}")
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
                llm_results=[],
                total_issues=0,
                detection_summary={}
            )
        
        print(f"      🔍 Analyzing field '{field_name}' (column: '{column_name}')")
        
        validation_results = []
        anomaly_results = []
        ml_results = []
        llm_results = []
        
        # Run validation if enabled and available
        if self.enable_validation:
            validator, validator_reporter = self._get_validator_components(field_name)
            if validator and validator_reporter:
                try:
                    validation_errors = validator.bulk_validate(df, column_name)
                    validation_report = validator_reporter.generate_report(validation_errors, df)
                    validation_results = [r for r in validation_report if r.get('probability', 1.0) >= self.validation_threshold]
                    print(f"         📝 Validation: {len(validation_results)} errors found")
                except Exception as e:
                    print(f"         ⚠️  Validation failed: {e}")
        
        # Run pattern-based anomaly detection if enabled and available
        if self.enable_pattern:
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
        
        # Run ML-based anomaly detection if enabled and available
        if self.enable_ml:
            ml_detector = self._get_ml_detector(field_name)
            if ml_detector:
                try:
                    # Initialize the ML detector
                    ml_detector.learn_patterns(df, column_name)
                    
                    # Process each row individually with the ML detector
                    ml_results_formatted = []
                    for idx, value in df[column_name].items():
                        anomaly_error = ml_detector._detect_anomaly(value)
                        if anomaly_error and anomaly_error.probability >= self.ml_threshold:
                            ml_results_formatted.append({
                                'row_index': idx,
                                'column_name': column_name,
                                'error_data': value,
                                'display_message': f"ML Anomaly: {anomaly_error.explanation}",
                                'probability': anomaly_error.probability,
                                'detection_type': 'ml_based'
                            })
                    
                    ml_results = ml_results_formatted
                    print(f"         🤖 ML anomalies: {len(ml_results)} anomalies found")
                except Exception as e:
                    print(f"         ⚠️  ML anomaly detection failed: {e}")
                
                # Force cleanup of ML detector immediately after use
                del ml_detector
                gc.collect()
        
        # Run LLM-based anomaly detection if enabled and available
        if self.enable_llm:
            llm_detector = self._get_llm_detector(field_name)
            if llm_detector:
                try:
                    # Initialize the LLM detector (loads trained model)
                    llm_detector.learn_patterns(df, column_name)
                    
                    # For bulk detection, we'll call the bulk_detect method
                    llm_anomalies = llm_detector.bulk_detect(df, column_name, self.batch_size, self.max_workers)
                    
                    llm_results_formatted = []
                    for anomaly_error in llm_anomalies:
                        if anomaly_error:
                            # LLM detector already applied the threshold, so any returned anomaly is valid
                            llm_results_formatted.append({
                                'row_index': anomaly_error.row_index,
                                'column_name': column_name,
                                'error_data': anomaly_error.anomaly_data,
                                'display_message': f"LLM Anomaly: {anomaly_error.explanation}",
                                'probability': anomaly_error.probability,
                                'detection_type': 'llm_based'
                            })
                    
                    llm_results = llm_results_formatted
                    print(f"         🤖 LLM anomalies: {len(llm_results)} anomalies found")
                except Exception as e:
                    print(f"         ⚠️  LLM anomaly detection failed: {e}")
                
                # Force cleanup of LLM detector immediately after use
                del llm_detector
                gc.collect()
        
        total_issues = len(validation_results) + len(anomaly_results) + len(ml_results) + len(llm_results)
        detection_summary = {
            "validation_errors": len(validation_results),
            "pattern_anomalies": len(anomaly_results),
            "ml_anomalies": len(ml_results),
            "llm_anomalies": len(llm_results)
        }
        
        return FieldDetectionResult(
            field_name=field_name,
            column_name=column_name,
            validation_results=validation_results,
            anomaly_results=anomaly_results,
            ml_results=ml_results,
            llm_results=llm_results,
            total_issues=total_issues,
            detection_summary=detection_summary
        )
    
    def classify_cells(self, df: pd.DataFrame, field_results: Dict[str, FieldDetectionResult]) -> List[CellClassification]:
        """
        Classify each cell based on detection results using either priority-based or weighted combination.
        
        This method routes to the appropriate classification approach based on the 
        use_weighted_combination flag set during initialization.
        
        Args:
            df: Original DataFrame
            field_results: Detection results for all fields
            
        Returns:
            List of cell classifications (excludes clean cells)
        """
        if self.use_weighted_combination:
            # Use performance-based weighted combination of detection methods
            return self._weighted_classify_cells(df, field_results)
        else:
            # Use traditional priority-based hierarchy (validation > pattern > ML > LLM)
            return self._priority_classify_cells(df, field_results)
    
    def _priority_classify_cells(self, df: pd.DataFrame, field_results: Dict[str, FieldDetectionResult]) -> List[CellClassification]:
        """
        Classify cells using priority-based approach: validation > pattern-based > ML-based > LLM-based.
        """
        print(f"   🎯 Classifying cells with PRIORITY-BASED approach: validation > pattern-based > ML-based > LLM-based")
        
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
                            'detected_value': detection.get('error_data'),
                            'error_code': detection.get('error_code')
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
                            'detected_value': detection.get('error_data'),
                            'error_code': detection.get('anomaly_code') or detection.get('error_code')
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
                            'detected_value': detection.get('error_data'),
                            'error_code': detection.get('error_code')
                        }
            
            # Add LLM-based anomaly results (lowest priority)
            for detection in result.llm_results:
                row_idx = detection.get('row_index')
                if row_idx is not None:
                    key = (row_idx, column_name)
                    if key not in detection_map or detection_map[key]['priority'] > 4:
                        detection_map[key] = {
                            'field_name': field_name,
                            'priority': 4,  # Lowest priority
                            'status': 'ANOMALY',
                            'detection_type': 'llm_based',
                            'message': detection.get('display_message', 'LLM-based anomaly'),
                            'confidence': detection.get('probability', 0.5),
                            'detected_value': detection.get('error_data'),
                            'error_code': detection.get('error_code')
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
                detected_value=detection_info['detected_value'],
                error_code=detection_info.get('error_code')
            )
            
            cell_classifications.append(classification)
        
        # Sort by row_index for easier navigation
        cell_classifications.sort(key=lambda x: (x.row_index, x.column_name))
        
        print(f"   ✅ Classified {len(cell_classifications)} cells with issues")
        return cell_classifications
    
    def _weighted_classify_cells(self, df: pd.DataFrame, field_results: Dict[str, FieldDetectionResult]) -> List[CellClassification]:
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
        
        if self.use_weighted_combination:
            print(f"   📊 Detection weights by field:")
            for field_name in field_results.keys():
                weights = self.get_field_detection_weights(field_name)
                weight_str = ", ".join([f"{method}: {weight:.2f}" for method, weight in weights.items()])
                print(f"       {field_name}: {weight_str}")
        
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
            gc.collect()
            
            print(f"      ✅ {field_name}: {result.total_issues} issues found, {len(field_classifications)} cells classified")
        
        # Print final summary
        total_issues = sum(len(result.validation_results) + len(result.anomaly_results) + len(result.ml_results) + len(result.llm_results) 
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