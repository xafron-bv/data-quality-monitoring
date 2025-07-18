import pandas as pd
import json
import os
from typing import List, Dict, Any, Union
import numpy as np
from collections import defaultdict

from anomaly_detectors.reporter_interface import AnomalyReporterInterface, MLAnomalyResult
from anomaly_detectors.anomaly_error import AnomalyError

class MLAnomalyReporter(AnomalyReporterInterface):
    """
    Implements the AnomalyReporterInterface for ML-based anomaly detection results.
    This reporter translates complex ML model outputs into human-readable messages.
    """

    def __init__(self, model_name: str, include_technical_details: bool = False):
        """
        Initialize the ML anomaly reporter.
        
        Args:
            model_name: The name of the ML model or pipeline used for detection
            include_technical_details: Whether to include detailed technical information in reports
        """
        self.model_name = model_name
        self.include_technical_details = include_technical_details
        
        # Load explanation templates
        templates_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "anomaly_detectors", "ml_explanation_templates.json"
        )
        
        try:
            with open(templates_path, 'r') as f:
                self.explanation_templates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default templates if file not found or invalid
            self.explanation_templates = {
                "default": "This value was detected as anomalous by the {model_name} model with a score of {score:.2f}.",
                "high_score": "This value is highly anomalous according to the {model_name} model (score: {score:.2f}).",
                "outlier": "This value is a statistical outlier, {z_score:.2f} standard deviations from the mean.",
                "cluster": "This value doesn't match the typical patterns for this category.",
                "feature_contribution": "The unusual aspects of this value are: {top_features}."
            }

    def _format_ml_explanation(self, result: MLAnomalyResult) -> str:
        """
        Create a human-readable explanation from ML model results.
        
        Args:
            result: The ML anomaly result object
            
        Returns:
            A string containing the explanation
        """
        # If the model already provided an explanation, use it
        if result.explanation:
            return result.explanation
            
        # Otherwise build an explanation based on available data
        explanations = []
        
        # Add base explanation based on score
        template = self.explanation_templates["high_score"] if result.probabiliy > 0.85 else self.explanation_templates["default"]
        explanations.append(template.format(model_name=self.model_name, score=result.probabiliy))
        
        # Add statistical explanation if available
        if result.probability_info and "z_score" in result.probability_info:
            z_score = abs(result.probability_info["z_score"])
            if z_score > 2:
                stat_template = self.explanation_templates["outlier"]
                explanations.append(stat_template.format(z_score=z_score))
        
        # Add feature contribution explanation if available
        if result.feature_contributions:
            # Get top 3 contributing features
            top_features = sorted(
                result.feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]
            
            if top_features:
                feature_list = ", ".join([f"{feat} ({score:.2f})" for feat, score in top_features])
                feature_template = self.explanation_templates["feature_contribution"]
                explanations.append(feature_template.format(top_features=feature_list))
        
        # Add cluster information if available
        if result.cluster_info and "cluster_name" in result.cluster_info:
            cluster_template = self.explanation_templates["cluster"]
            explanations.append(cluster_template)
            
        return " ".join(explanations)

    def generate_report(self, 
                      anomaly_results: Union[List[AnomalyError], List[MLAnomalyResult]], 
                      original_df: pd.DataFrame,
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate human-readable reports from anomaly detection results.
        
        Args:
            anomaly_results: List of anomaly detection results (ML or rule-based)
            original_df: Original DataFrame for context
            threshold: Threshold for reporting anomalies
            
        Returns:
            List of report dictionaries
        """
        reports = []
        
        for result in anomaly_results:
            if isinstance(result, MLAnomalyResult):
                # Process ML-based result
                if result.probabiliy < threshold:
                    continue  # Skip if below threshold
                
                # Generate explanation
                explanation = self._format_ml_explanation(result)
                
                # Build report
                report = {
                    "row_index": result.row_index,
                    "column_name": result.column_name,
                    "value": result.value,
                    "display_message": explanation,
                    "probabiliy": result.probabiliy,
                    "is_ml_based": True
                }
                
                # Add technical details if requested
                if self.include_technical_details:
                    report["technical_details"] = {
                        "feature_contributions": result.feature_contributions,
                        "probability_info": result.probability_info,
                        "cluster_info": result.cluster_info,
                        "nearest_neighbors": [
                            {"row_index": idx, "distance": dist} 
                            for idx, dist in result.nearest_neighbors[:5]
                        ] if result.nearest_neighbors else []
                    }
                
                reports.append(report)
                
            elif isinstance(result, AnomalyError):
                # Process rule-based result
                probability = result.probability
                if probability < threshold:
                    continue  # Skip if below threshold
                
                # Build report
                report = {
                    "row_index": result.row_index,
                    "column_name": result.column_name,
                    "value": result.anomaly_data,
                    "display_message": f"Anomaly detected: {result.anomaly_type}",
                    "probabiliy": probability,
                    "explanation": json.dumps(result.details) if result.details else None,
                    "is_ml_based": False
                }
                
                reports.append(report)
        
        return reports
