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
        """Initialize the anomaly detector with fixed domain knowledge."""
        self.field_name = "material"  # The type of field this detector validates
        self.material_regex = re.compile(r'(\d+(?:\.\d+)?%)\s+([A-Za-z]+)')
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?%?|[a-zA-Z]+|[^a-zA-Z0-9\s%])')
        
        # Fixed domain knowledge: known textile materials (case-insensitive)
        self.known_materials = {
            # Natural fibers
            'cotton', 'wool', 'silk', 'linen', 'hemp', 'bamboo', 'cashmere', 'mohair', 
            'alpaca', 'angora', 'vicuna', 'merino', 'jersey',
            # Synthetic fibers  
            'polyester', 'nylon', 'acrylic', 'spandex', 'elastane', 'lycra', 'polyurethane',
            'rayon', 'viscose', 'modal', 'tencel', 'microfiber', 'fleece',
            # Blended materials
            'poly', 'acetate', 'triacetate', 'metallic', 'lurex',
            # Common variations
            'cottn', 'cotton', 'woolen', 'polyamide', 'polypropylene'
        }
        
        # Known non-textile materials that should be anomalous
        self.non_textile_materials = {
            # Construction materials
            'concrete', 'steel', 'aluminum', 'wood', 'plastic', 'glass', 'brick',
            'mortar', 'titanium', 'ceramic', 'granite', 'marble', 'diamond', 'carbon',
            # Food materials  
            'bread', 'cheese', 'meat', 'fruit', 'vegetable', 'flour', 'sugar',
            'chocolate', 'butter', 'milk', 'cream', 'honey', 'salt', 'pepper',
            # Electronics
            'silicon', 'copper', 'gold', 'silver', 'platinum', 'circuit', 'chip',
            # Other inappropriate materials
            'paper', 'cardboard', 'rubber', 'foam', 'gel', 'liquid', 'gas'
        }
    
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
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in material composition based on fixed domain rules.
        
        This method checks for:
        1. Non-textile materials (construction, food, electronics, etc.)
        2. Invalid material format patterns
        3. Suspicious percentage compositions
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
        
        # Extract material info
        materials = self._extract_material_info(value)
        
        # Check if no materials were extracted (invalid format)
        if not materials:
            # Check for completely non-standard format
            text_lower = value.lower().strip()
            if any(non_textile in text_lower for non_textile in self.non_textile_materials):
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNCOMMON_MATERIAL,
                    probability=0.95,
                    details={
                        "material": value,
                        "reason": "Contains non-textile material",
                        "detected_non_textile": [mat for mat in self.non_textile_materials if mat in text_lower]
                    }
                )
            return None
        
        # Check each material for domain violations
        for item in materials:
            material = item["material"].lower()
            percentage = item["percentage"]
            
            # Check for non-textile materials
            if material in self.non_textile_materials:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNCOMMON_MATERIAL,
                    probability=0.95,
                    details={
                        "material": material,
                        "reason": "Non-textile material detected",
                        "category": "construction/food/electronics"
                    }
                )
            
            # Check for unknown materials (not in known textile set)
            if material not in self.known_materials:
                # Check if it's a partial match or typo of known material
                close_matches = [known for known in self.known_materials 
                               if known.startswith(material[:3]) or material.startswith(known[:3])]
                
                probability = 0.85 if close_matches else 0.95
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNCOMMON_MATERIAL,
                    probability=probability,
                    details={
                        "material": material,
                        "reason": "Unknown material not in textile database",
                        "possible_matches": close_matches[:3] if close_matches else []
                    }
                )
            
            # Check for unusual percentages
            if percentage <= 0 or percentage > 100:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNUSUAL_PERCENTAGE,
                    probability=0.90,
                    details={
                        "material": material,
                        "percentage": percentage,
                        "reason": "Invalid percentage value"
                    }
                )
        
        # Check total percentage sum
        total_percentage = sum(item["percentage"] for item in materials)
        if abs(total_percentage - 100.0) > 5.0:  # Allow 5% tolerance
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNEXPECTED_COMPOSITION,
                probability=0.80,
                details={
                    "total_percentage": total_percentage,
                    "reason": "Material percentages don't sum to ~100%",
                    "materials": [f"{item['percentage']}% {item['material']}" for item in materials]
                }
            )
        
        # No anomalies detected
        return None
