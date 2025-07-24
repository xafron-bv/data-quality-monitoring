import pandas as pd
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
import re
import string

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for color names.
    This detector identifies anomalies in color names by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        RARE_COLOR = "RARE_COLOR"
        UNUSUAL_CAPITALIZATION = "UNUSUAL_CAPITALIZATION"
        UNUSUAL_CHARACTER_SET = "UNUSUAL_CHARACTER_SET"
    
    def __init__(self):
        """Initialize the anomaly detector with fixed domain knowledge."""
        # Fixed domain knowledge: known color names (case-insensitive)
        self.known_colors = {
            # Basic colors
            'black', 'white', 'grey', 'gray', 'red', 'blue', 'green', 'yellow', 
            'orange', 'purple', 'pink', 'brown', 'beige', 'cream', 'navy',
            # Fashion colors
            'marine', 'khaki', 'olive', 'burgundy', 'maroon', 'teal', 'turquoise',
            'coral', 'salmon', 'mint', 'lavender', 'rose', 'gold', 'silver',
            'bronze', 'copper', 'ivory', 'pearl', 'champagne', 'nude', 'taupe',
            # Darker shades
            'charcoal', 'slate', 'graphite', 'ebony', 'jet', 'onyx', 'raven',
            # Lighter shades  
            'vanilla', 'snow', 'cotton', 'milk', 'powder', 'chalk', 'ash',
            # Vibrant colors
            'crimson', 'scarlet', 'ruby', 'emerald', 'sapphire', 'cobalt',
            'azure', 'indigo', 'violet', 'magenta', 'fuchsia', 'lime', 'citrus',
            # Earthy tones
            'sand', 'clay', 'rust', 'copper', 'bronze', 'mahogany', 'chestnut',
            'walnut', 'oak', 'maple', 'cedar', 'pine', 'moss', 'sage', 'forest',
            # Seasonal colors
            'autumn', 'winter', 'spring', 'summer', 'seasonal'
        }
        
        # Known non-color terms that should be anomalous
        self.non_color_terms = {
            # Food terms
            'mustard', 'ketchup', 'avocado', 'chocolate', 'vanilla', 'caramel',
            'honey', 'maple', 'lemon', 'grape', 'apple', 'orange', 'berry',
            # Construction materials  
            'concrete', 'steel', 'aluminum', 'brass', 'iron', 'chrome',
            # Inappropriate terms
            'electronic', 'digital', 'virtual', 'synthetic', 'artificial',
            'video', 'computer', 'machine', 'robot', 'cyber'
        }
    
    def _normalize_color(self, color: str) -> str:
        """Normalize a color name for comparison."""
        if not isinstance(color, str):
            return ""
        return color.lower().strip()
    

    
    def _get_capitalization_pattern(self, color: str) -> str:
        """Generate a pattern that represents the capitalization style."""
        if not isinstance(color, str) or not color:
            return ""
        
        result = ""
        for word in color.split():
            if word:
                if word[0].isupper():
                    result += "C"  # Capitalized
                else:
                    result += "l"  # lowercase
        return result
    
    def _get_character_set(self, color: str) -> str:
        """Generate a pattern that represents the character types used."""
        if not isinstance(color, str) or not color:
            return ""
        
        has_alpha = bool(re.search(r'[a-zA-Z]', color))
        has_digit = bool(re.search(r'\d', color))
        has_space = ' ' in color
        has_punct = bool(re.search(f'[{re.escape(string.punctuation)}]', color))
        
        result = ""
        if has_alpha: result += "A"
        if has_digit: result += "D"
        if has_space: result += "S"
        if has_punct: result += "P"
        
        return result if result else "X"  # X for empty/invalid
        
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in color names based on fixed domain rules.
        
        This method checks for:
        1. Non-color terms (food, construction, electronics, etc.)
        2. Unknown colors not in fashion database
        3. Invalid format patterns
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
        
        norm_color = self._normalize_color(value).lower()
        
        # Check for non-color terms that should be anomalous
        for non_color in self.non_color_terms:
            if non_color in norm_color:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.RARE_COLOR,
                    probability=0.95,
                    details={
                        "color": value,
                        "reason": "Contains non-color term",
                        "detected_non_color": non_color,
                        "category": "food/construction/electronics"
                    }
                )
        
        # Check if color is in known fashion colors
        if norm_color not in self.known_colors:
            # Check for partial matches (possible typos)
            close_matches = [known for known in self.known_colors 
                           if known.startswith(norm_color[:3]) or norm_color.startswith(known[:3])]
            
            probability = 0.80 if close_matches else 0.90
            return AnomalyError(
                anomaly_type=self.ErrorCode.RARE_COLOR,
                probability=probability,
                details={
                    "color": norm_color,
                    "reason": "Unknown color not in fashion database",
                    "possible_matches": close_matches[:3] if close_matches else []
                }
            )
        
        # Check for unusual character patterns (too many special characters)
        if re.search(r'[^\w\s-]', value):  # Allow only letters, numbers, spaces, hyphens
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_CHARACTER_SET,
                probability=0.85,
                details={
                    "color": value,
                    "reason": "Contains unusual special characters",
                    "invalid_chars": re.findall(r'[^\w\s-]', value)
                }
            )
        
        # No anomalies detected
        return None
