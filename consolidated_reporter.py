#!/usr/bin/env python3
"""
Consolidated Reporter

Generates JSON reports compatible with data_quality_viewer.html from comprehensive
detection results. Creates unified reports that show all errors, anomalies, and
ML-detected issues across all fields in a format that the viewer can understand.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from comprehensive_detector import CellClassification, FieldDetectionResult


def clean_value_for_json(value: Any) -> Any:
    """Clean a value to ensure JSON compatibility."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and (value != value):  # Additional NaN check
        return None
    try:
        # Test if it's JSON serializable
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def create_viewer_compatible_report(cell_classifications: List[CellClassification],
                                  field_results: Dict[str, FieldDetectionResult],
                                  sample_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a report compatible with data_quality_viewer.html.
    
    Args:
        cell_classifications: List of cell classifications from comprehensive detection
        field_results: Detection results for each field
        sample_metadata: Optional metadata about the sample (e.g., injection info)
        
    Returns:
        Dict in the format expected by data_quality_viewer.html
    """
    
    # Separate classifications by detection type for viewer compatibility
    errors = []
    anomalies = []
    ml_issues = []
    
    for classification in cell_classifications:
        # Create base entry
        entry = {
            "row_index": classification.row_index,
            "column_name": classification.column_name,
            "display_message": clean_value_for_json(classification.message),
            "probability": clean_value_for_json(classification.confidence),
            "error_data": clean_value_for_json(classification.detected_value),
            "severity": classification.status.lower(),  # "error" or "anomaly"
            "type": classification.detection_type
        }
        
        # Route to appropriate category based on detection type
        if classification.detection_type == "validation":
            errors.append(entry)
        elif classification.detection_type == "pattern_based":
            anomalies.append(entry)
        elif classification.detection_type == "ml_based":
            ml_issues.append(entry)
        else:
            # Fallback: route based on status
            if classification.status == "ERROR":
                errors.append(entry)
            else:
                anomalies.append(entry)
    
    # Calculate summary statistics
    total_errors = len(errors)
    total_anomalies = len(anomalies)
    total_ml_issues = len(ml_issues)
    total_issues = total_errors + total_anomalies + total_ml_issues
    
    # Count affected rows
    affected_rows = len(set(
        classification.row_index for classification in cell_classifications
    ))
    
    # Create the cell_coordinates structure expected by the viewer
    cell_coordinates = {
        "errors": errors,
        "anomalies": anomalies,
        "ml_issues": ml_issues,
        "summary": {
            "total_errors": total_errors,
            "total_anomalies": total_anomalies,
            "total_ml_issues": total_ml_issues,
            "total_issues": total_issues,
            "affected_rows": affected_rows
        }
    }
    
    # Create the full report structure
    report = {
        "cell_coordinates": cell_coordinates
    }
    
    # Add sample metadata if provided
    if sample_metadata:
        report["sample_info"] = clean_value_for_json(sample_metadata)
    
    # Add field summaries
    field_summaries = {}
    for field_name, result in field_results.items():
        field_summaries[field_name] = {
            "column_name": result.column_name,
            "total_issues": result.total_issues,
            "detection_summary": result.detection_summary
        }
    
    report["field_summaries"] = field_summaries
    
    return report


def create_detailed_analysis_report(cell_classifications: List[CellClassification],
                                  field_results: Dict[str, FieldDetectionResult],
                                  sample_df: pd.DataFrame,
                                  injection_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """
    Create a detailed analysis report with additional insights.
    
    Args:
        cell_classifications: Cell classifications
        field_results: Field detection results
        sample_df: The analyzed DataFrame
        injection_metadata: Optional injection metadata for evaluation
        
    Returns:
        Detailed analysis report
    """
    
    # Basic statistics
    total_rows = len(sample_df)
    total_columns = len(sample_df.columns)
    analyzed_fields = len(field_results)
    
    # Detection method availability
    methods_by_field = {}
    for field_name, result in field_results.items():
        methods = []
        if result.detection_summary.get("validation_errors", 0) >= 0:
            methods.append("validation")
        if result.detection_summary.get("pattern_anomalies", 0) >= 0:
            methods.append("pattern_based")
        if result.detection_summary.get("ml_anomalies", 0) >= 0:
            methods.append("ml_based")
        methods_by_field[field_name] = methods
    
    # Issue distribution by field
    issues_by_field = {}
    for classification in cell_classifications:
        field_name = classification.field_name
        if field_name not in issues_by_field:
            issues_by_field[field_name] = {
                "validation": 0,
                "pattern_based": 0,
                "ml_based": 0,
                "total": 0
            }
        
        issues_by_field[field_name][classification.detection_type] += 1
        issues_by_field[field_name]["total"] += 1
    
    # Issue distribution by row
    issues_by_row = {}
    for classification in cell_classifications:
        row_idx = classification.row_index
        if row_idx not in issues_by_row:
            issues_by_row[row_idx] = 0
        issues_by_row[row_idx] += 1
    
    # Performance evaluation if injection metadata is available
    performance_analysis = None
    if injection_metadata:
        performance_analysis = analyze_detection_performance(
            cell_classifications, injection_metadata
        )
    
    return {
        "dataset_info": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "analyzed_fields": analyzed_fields,
            "available_fields": list(field_results.keys())
        },
        "detection_capabilities": {
            "methods_by_field": methods_by_field,
            "total_fields_with_validation": sum(1 for methods in methods_by_field.values() if "validation" in methods),
            "total_fields_with_pattern_detection": sum(1 for methods in methods_by_field.values() if "pattern_based" in methods),
            "total_fields_with_ml_detection": sum(1 for methods in methods_by_field.values() if "ml_based" in methods)
        },
        "issue_distribution": {
            "by_field": issues_by_field,
            "by_row": dict(sorted(issues_by_row.items())),
            "most_problematic_fields": sorted(
                issues_by_field.items(), 
                key=lambda x: x[1]["total"], 
                reverse=True
            )[:5],
            "most_problematic_rows": sorted(
                issues_by_row.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        },
        "performance_analysis": performance_analysis
    }


def analyze_detection_performance(cell_classifications: List[CellClassification],
                                injection_metadata: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze detection performance by comparing classifications with injection metadata.
    
    Args:
        cell_classifications: Detected issues
        injection_metadata: Known injected issues
        
    Returns:
        Performance analysis metrics
    """
    
    # Create sets of detected and injected issues for comparison
    detected_issues = set()
    for classification in cell_classifications:
        key = (classification.row_index, classification.column_name)
        detected_issues.add(key)
    
    injected_issues = set()
    injected_by_type = {"error": set(), "anomaly": set()}
    
    for field_name, injections in injection_metadata.items():
        for injection in injections:
            row_idx = injection["row_index"]
            # Need to map field_name to column_name for comparison
            # This is a simplified approach - in practice, we'd need field_mapper
            column_name = field_name  # Simplified assumption
            key = (row_idx, column_name)
            injected_issues.add(key)
            injected_by_type[injection["injection_type"]].add(key)
    
    # Calculate metrics
    true_positives = detected_issues & injected_issues
    false_positives = detected_issues - injected_issues
    false_negatives = injected_issues - detected_issues
    
    total_detected = len(detected_issues)
    total_injected = len(injected_issues)
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    # Calculate performance metrics
    precision = tp_count / total_detected if total_detected > 0 else 0
    recall = tp_count / total_injected if total_injected > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count
        },
        "injection_summary": {
            "total_injected": total_injected,
            "total_detected": total_detected,
            "errors_injected": len(injected_by_type["error"]),
            "anomalies_injected": len(injected_by_type["anomaly"])
        },
        "performance_interpretation": {
            "precision_meaning": f"Of {total_detected} flagged issues, {tp_count} were actual injected issues ({precision*100:.1f}%)",
            "recall_meaning": f"Of {total_injected} injected issues, {tp_count} were successfully detected ({recall*100:.1f}%)",
            "f1_interpretation": get_f1_interpretation(f1_score)
        }
    }


def get_f1_interpretation(f1_score: float) -> str:
    """Get human-readable interpretation of F1 score."""
    if f1_score >= 0.9:
        return "Excellent detection performance"
    elif f1_score >= 0.7:
        return "Good detection performance"
    elif f1_score >= 0.5:
        return "Moderate detection performance"
    elif f1_score >= 0.3:
        return "Poor detection performance"
    else:
        return "Very poor detection performance"


def save_consolidated_reports(cell_classifications: List[CellClassification],
                            field_results: Dict[str, FieldDetectionResult],
                            sample_df: pd.DataFrame,
                            output_dir: str,
                            injection_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                            sample_name: str = "comprehensive_analysis") -> Dict[str, str]:
    """
    Save all consolidated reports to files.
    
    Args:
        cell_classifications: Cell classifications
        field_results: Field detection results
        sample_df: Analyzed DataFrame
        output_dir: Output directory
        injection_metadata: Optional injection metadata
        sample_name: Base name for output files
        
    Returns:
        Dict with paths to created files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create viewer-compatible report
    viewer_report = create_viewer_compatible_report(
        cell_classifications, field_results, 
        sample_metadata={"name": sample_name, "total_rows": len(sample_df)}
    )
    
    # Create detailed analysis report
    detailed_report = create_detailed_analysis_report(
        cell_classifications, field_results, sample_df, injection_metadata
    )
    
    # Save viewer-compatible report (main file for visualization)
    viewer_path = os.path.join(output_dir, f"{sample_name}_viewer_report.json")
    with open(viewer_path, 'w', encoding='utf-8') as f:
        json.dump(viewer_report, f, indent=2, ensure_ascii=False)
    
    # Save detailed analysis report
    detailed_path = os.path.join(output_dir, f"{sample_name}_detailed_analysis.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)
    
    # Save cell coordinates specifically (for backward compatibility)
    cell_coords_path = os.path.join(output_dir, f"{sample_name}_cell_coordinates.json")
    with open(cell_coords_path, 'w', encoding='utf-8') as f:
        json.dump(viewer_report["cell_coordinates"], f, indent=2, ensure_ascii=False)
    
    # Save summary for quick reference
    summary = {
        "sample_name": sample_name,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_rows": len(sample_df),
        "dataset_columns": len(sample_df.columns),
        "analyzed_fields": len(field_results),
        "total_issues_detected": len(cell_classifications),
        "affected_rows": len(set(c.row_index for c in cell_classifications)),
        "issues_by_type": {
            "validation_errors": sum(1 for c in cell_classifications if c.detection_type == "validation"),
            "pattern_anomalies": sum(1 for c in cell_classifications if c.detection_type == "pattern_based"),
            "ml_anomalies": sum(1 for c in cell_classifications if c.detection_type == "ml_based")
        },
        "files_created": {
            "viewer_report": f"{sample_name}_viewer_report.json",
            "detailed_analysis": f"{sample_name}_detailed_analysis.json",
            "cell_coordinates": f"{sample_name}_cell_coordinates.json"
        }
    }
    
    summary_path = os.path.join(output_dir, f"{sample_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return {
        "viewer_report": viewer_path,
        "detailed_analysis": detailed_path,
        "cell_coordinates": cell_coords_path,
        "summary": summary_path
    } 