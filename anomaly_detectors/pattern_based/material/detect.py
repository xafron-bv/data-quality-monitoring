import pandas as pd
import re
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
import numpy as np

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for material composition strings.
    This detector identifies anomalies in material descriptions by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        UNCOMMON_MATERIAL = "UNCOMMON_MATERIAL"
        UNUSUAL_PERCENTAGE = "UNUSUAL_PERCENTAGE"
        UNEXPECTED_COMPOSITION = "UNEXPECTED_COMPOSITION"
        UNUSUAL_MATERIAL_COMBINATION = "UNUSUAL_MATERIAL_COMBINATION"
        INCONSISTENT_FORMAT = "INCONSISTENT_FORMAT"
    
    def __init__(self):
        """Initialize the anomaly detector with state variables."""
        self.field_name = "material"  # The type of field this detector validates
        self.material_counts = Counter()
        self.percentage_distribution = defaultdict(list)
        self.common_material_combinations = defaultdict(int)
        self.format_patterns = Counter()
        self.total_records = 0
        self.material_regex = re.compile(r'(\d+(?:\.\d+)?%)\s+([A-Za-z]+)')
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])')
    
    def _extract_material_info(self, text: str) -> List[Dict[str, str]]:
        """Extract percentages and materials from a material composition string."""
        if not isinstance(text, str):
            return []
        
        matches = self.material_regex.findall(text)
        return [
            {"percentage": float(pct.rstrip('%')), "material": material.lower()}
            for pct, material in matches
        ]
    
    def _get_format_pattern(self, text: str) -> str:
        """Convert a material string to a format pattern for comparison."""
        if not isinstance(text, str):
            return ""
        
        # Replace digits with D, letters with L, and keep special chars
        pattern = re.sub(r'\d+', 'D', text)
        pattern = re.sub(r'[a-zA-Z]+', 'L', pattern)
        return pattern
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from material composition data.
        
        This method analyzes the material column to learn:
        1. Common material types
        2. Typical percentage distributions for each material
        3. Common material combinations
        4. Standard format patterns
        """
        self.total_records = len(df)
        valid_data = df[column_name].dropna()
        
        # Learn common materials and their percentages
        for text in valid_data:
            if not isinstance(text, str):
                continue
                
            # Extract material info
            materials = self._extract_material_info(text)
            
            # Count materials
            for item in materials:
                self.material_counts[item["material"]] += 1
                self.percentage_distribution[item["material"]].append(item["percentage"])
            
            # Track material combinations
            material_set = frozenset(item["material"] for item in materials)
            if len(material_set) > 1:
                self.common_material_combinations[material_set] += 1
            
            # Analyze format patterns
            format_pattern = self._get_format_pattern(text)
            self.format_patterns[format_pattern] += 1
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in material composition based on learned patterns.
        
        This method checks for:
        1. Uncommon materials
        2. Unusual percentage values for common materials
        3. Uncommon combinations of materials
        4. Inconsistent format patterns
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
        
        # Extract material info
        materials = self._extract_material_info(value)
        format_pattern = self._get_format_pattern(value)
        
        # Skip if no materials extracted (might be incorrect format)
        if not materials:
            # Check if the format is unusual
            if self.format_patterns and self.format_patterns[format_pattern] < self.total_records * 0.05:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.INCONSISTENT_FORMAT,
                    probability=0.7,
                    details={
                        "format": format_pattern,
                        "common_formats": [f for f, c in self.format_patterns.most_common(3)]
                    }
                )
            return None
        
        # Check for uncommon materials
        for item in materials:
            material = item["material"]
            percentage = item["percentage"]
            
            material_frequency = self.material_counts.get(material, 0)
            if 0 < material_frequency < self.total_records * 0.05:
                # Uncommon material
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNCOMMON_MATERIAL,
                    probability=min(0.95, 1.0 - (material_frequency / self.total_records)),
                    details={
                        "material": material,
                        "frequency": material_frequency,
                        "total_records": self.total_records
                    }
                )
            
            # Check for unusual percentages
            if material in self.percentage_distribution and len(self.percentage_distribution[material]) > 5:
                percentages = self.percentage_distribution[material]
                mean_pct = np.mean(percentages)
                std_pct = max(1, np.std(percentages))  # Ensure minimum std to avoid false positives
                
                if abs(percentage - mean_pct) > 2 * std_pct:
                    # Unusual percentage
                    z_score = (percentage - mean_pct) / std_pct if std_pct > 0 else 0
                    return AnomalyError(
                        anomaly_type=self.ErrorCode.UNUSUAL_PERCENTAGE,
                        probability=min(0.9, abs(z_score) / 5),  # Cap at 0.9
                        details={
                            "material": material,
                            "percentage": percentage,
                            "typical_range": f"{mean_pct-std_pct:.1f}% to {mean_pct+std_pct:.1f}%",
                            "z_score": z_score
                        }
                    )
        
        # Check for unusual material combinations
        material_set = frozenset(item["material"] for item in materials)
        if len(material_set) > 1:
            combination_frequency = self.common_material_combinations.get(material_set, 0)
            if 0 < combination_frequency < self.total_records * 0.05:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNUSUAL_MATERIAL_COMBINATION,
                    probability=min(0.9, 1.0 - (combination_frequency / self.total_records)),
                    details={
                        "materials": sorted(list(material_set)),
                        "frequency": combination_frequency,
                        "total_records": self.total_records
                    }
                )
        
        # No anomalies detected
        return None
