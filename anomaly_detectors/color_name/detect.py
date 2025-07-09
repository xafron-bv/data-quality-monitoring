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
        """Initialize the anomaly detector with state variables."""
        self.color_counts = Counter()
        self.capitalization_patterns = Counter()
        self.character_set_patterns = Counter()
        self.total_colors = 0
    
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
        
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from color name data.
        
        This method analyzes the color column to learn:
        1. Common color names
        2. Capitalization patterns
        3. Character sets used
        """
        valid_colors = df[column_name].dropna()
        self.total_colors = len(valid_colors)
        
        for color in valid_colors:
            if not isinstance(color, str) or not color.strip():
                continue
                
            # Count normalized colors
            norm_color = self._normalize_color(color)
            self.color_counts[norm_color] += 1
            
            # Analyze capitalization
            cap_pattern = self._get_capitalization_pattern(color)
            self.capitalization_patterns[cap_pattern] += 1
            
            # Analyze character set
            char_set = self._get_character_set(color)
            self.character_set_patterns[char_set] += 1
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in color names based on learned patterns.
        
        This method checks for:
        1. Rare color names
        2. Inconsistent capitalization
        3. Unusual character sets
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
        
        norm_color = self._normalize_color(value)
        cap_pattern = self._get_capitalization_pattern(value)
        char_set = self._get_character_set(value)
        
        # Check for rare colors
        color_frequency = self.color_counts.get(norm_color, 0)
        if 0 < color_frequency < self.total_colors * 0.03:  # Less than 3% frequency
            return AnomalyError(
                anomaly_type=self.ErrorCode.RARE_COLOR,
                confidence=min(0.9, 1.0 - (color_frequency / self.total_colors)),
                details={
                    "color": norm_color,
                    "frequency": color_frequency,
                    "total_colors": self.total_colors
                }
            )
        
        # Check for inconsistent capitalization
        if self.capitalization_patterns:
            common_cap = self.capitalization_patterns.most_common(1)[0][0]
            cap_frequency = self.capitalization_patterns.get(cap_pattern, 0)
            
            if cap_frequency < self.total_colors * 0.1 and cap_pattern != common_cap:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNUSUAL_CAPITALIZATION,
                    confidence=min(0.7, 1.0 - (cap_frequency / self.total_colors)),
                    details={
                        "capitalization": cap_pattern,
                        "common_pattern": common_cap,
                        "frequency": cap_frequency,
                        "total_colors": self.total_colors
                    }
                )
        
        # Check for unusual character sets
        if self.character_set_patterns:
            common_chars = self.character_set_patterns.most_common(1)[0][0]
            chars_frequency = self.character_set_patterns.get(char_set, 0)
            
            if chars_frequency < self.total_colors * 0.05 and char_set != common_chars:
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNUSUAL_CHARACTER_SET,
                    confidence=min(0.85, 1.0 - (chars_frequency / self.total_colors)),
                    details={
                        "character_set": char_set,
                        "common_set": common_chars,
                        "frequency": chars_frequency,
                        "total_colors": self.total_colors
                    }
                )
        
        # No anomalies detected
        return None
