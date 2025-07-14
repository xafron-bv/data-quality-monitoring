import pandas as pd
import re
from enum import Enum
from typing import Dict, Any, Optional, List, Set
from collections import Counter, defaultdict

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for textile care instructions.
    This detector identifies anomalies in care instruction text by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        UNUSUAL_LENGTH = "UNUSUAL_LENGTH"
        UNUSUAL_INSTRUCTION_SET = "UNUSUAL_INSTRUCTION_SET"
        UNUSUAL_STRUCTURE = "UNUSUAL_STRUCTURE"
        UNCOMMON_PATTERN = "UNCOMMON_PATTERN"
    
    def __init__(self):
        """Initialize the anomaly detector with state variables."""
        self.length_distribution = []  # Track the distribution of instruction lengths
        self.instruction_sets = Counter()  # Track common instruction sets
        self.delimiter_patterns = Counter()  # Track common delimiter patterns
        self.structure_patterns = Counter()  # Track structural patterns
        self.temperature_patterns = Counter()  # Track temperature mention patterns
        self.common_phrases = Counter()  # Track common phrases
        self.total_instructions = 0
    
    def _extract_instructions(self, text: str) -> Set[str]:
        """Extract individual instructions from care instruction text."""
        if not isinstance(text, str):
            return set()
            
        # Common delimiters: period, dot with space, comma, semicolon, etc.
        # Try multiple ways to split the text into instructions
        for delimiter in ['. ', '.', ', ', ',', ';', ' / ', '/']:
            if delimiter in text:
                instructions = {item.strip().upper() for item in text.split(delimiter) if item.strip()}
                if instructions:
                    return instructions
        
        # If no delimiter was found, return the whole text as a single instruction
        return {text.strip().upper()} if text.strip() else set()
    
    def _extract_structure_pattern(self, text: str) -> str:
        """Generate a structural pattern for the care instructions."""
        if not isinstance(text, str):
            return ""
            
        # Replace specific patterns
        pattern = text.upper()
        # Replace temperature patterns like "30°C", "40C", "30 C", etc.
        pattern = re.sub(r'\d+\s*[°]?\s*C', 'TEMP', pattern)
        # Replace general numbers
        pattern = re.sub(r'\d+', '#', pattern)
        # Create a fingerprint of the instruction types
        if "WAS" in pattern or "WASH" in pattern:
            pattern = "W"
        if "BLEACH" in pattern or "BLEK" in pattern:
            pattern += "B"
        if "DRY" in pattern or "DROGER" in pattern:
            pattern += "D"
        if "IRON" in pattern or "STRIJK" in pattern:
            pattern += "I"
        if "CLEAN" in pattern or "REINIG" in pattern:
            pattern += "C"
            
        return pattern
    
    def _get_delimiter_pattern(self, text: str) -> str:
        """Identify the delimiter pattern used in the care instructions."""
        if not isinstance(text, str):
            return ""
            
        for delimiter in ['. ', '.', ', ', ',', ';', ' / ', '/']:
            if delimiter in text:
                return f"DELIM:{delimiter}"
        return "DELIM:NONE"
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from care instruction data.
        
        This method analyzes the care instructions column to identify:
        1. Typical instruction lengths
        2. Common instruction sets
        3. Common delimiter and structure patterns
        4. Common phrases and temperature patterns
        """
        valid_instructions = df[column_name].dropna()
        self.total_instructions = len(valid_instructions)
        
        for text in valid_instructions:
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Track length
            self.length_distribution.append(len(text))
            
            # Extract and count instruction sets
            instructions = self._extract_instructions(text)
            self.instruction_sets[frozenset(instructions)] += 1
            
            # Track delimiter patterns
            delimiter_pattern = self._get_delimiter_pattern(text)
            self.delimiter_patterns[delimiter_pattern] += 1
            
            # Track structure patterns
            structure_pattern = self._extract_structure_pattern(text)
            self.structure_patterns[structure_pattern] += 1
            
            # Track temperature patterns
            temp_matches = re.findall(r'\d+\s*[°]?\s*C', text.upper())
            for match in temp_matches:
                self.temperature_patterns[match.strip()] += 1
            
            # Track common phrases
            phrases = re.findall(r'[A-Z]+\s+[A-Z]+(?:\s+[A-Z]+)?', text.upper())
            for phrase in phrases:
                if len(phrase) > 4:  # Only track substantial phrases
                    self.common_phrases[phrase.strip()] += 1
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in care instructions based on learned patterns.
        
        This method checks for:
        1. Unusual instruction length
        2. Unusual instruction sets
        3. Uncommon structure patterns
        4. Unusual delimiter patterns
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
            
        # Check for unusual length
        if self.length_distribution:
            length = len(value)
            sorted_lengths = sorted(self.length_distribution)
            q1_idx = int(len(sorted_lengths) * 0.25)
            q3_idx = int(len(sorted_lengths) * 0.75)
            q1 = sorted_lengths[q1_idx]
            q3 = sorted_lengths[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if length < lower_bound or length > upper_bound:
                # This is unusually short or long
                return AnomalyError(
                    anomaly_type=self.ErrorCode.UNUSUAL_LENGTH,
                    anomaly_score=min(0.8, abs(length - (q1 + q3)/2) / ((q3 - q1) * 2)),
                    details={
                        "length": length,
                        "typical_range": f"{q1} to {q3} characters"
                    }
                )
        
        # Check for unusual instruction set
        instructions = self._extract_instructions(value)
        instruction_set = frozenset(instructions)
        set_frequency = self.instruction_sets.get(instruction_set, 0)
        
        if instructions and self.total_instructions > 10 and 0 < set_frequency < self.total_instructions * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_INSTRUCTION_SET,
                anomaly_score=min(0.9, 1.0 - (set_frequency / self.total_instructions)),
                details={
                    "instructions": list(instructions),
                    "frequency": set_frequency,
                    "total_instructions": self.total_instructions
                }
            )
        
        # Check for unusual structure
        structure_pattern = self._extract_structure_pattern(value)
        structure_frequency = self.structure_patterns.get(structure_pattern, 0)
        
        if structure_pattern and self.total_instructions > 10 and 0 < structure_frequency < self.total_instructions * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_STRUCTURE,
                anomaly_score=min(0.85, 1.0 - (structure_frequency / self.total_instructions)),
                details={
                    "structure": structure_pattern,
                    "frequency": structure_frequency,
                    "common_structures": [s for s, _ in self.structure_patterns.most_common(3)]
                }
            )
        
        # Check for uncommon delimiter pattern
        delimiter_pattern = self._get_delimiter_pattern(value)
        delimiter_frequency = self.delimiter_patterns.get(delimiter_pattern, 0)
        
        if self.total_instructions > 10 and 0 < delimiter_frequency < self.total_instructions * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNCOMMON_PATTERN,
                anomaly_score=min(0.75, 1.0 - (delimiter_frequency / self.total_instructions)),
                details={
                    "pattern": delimiter_pattern,
                    "frequency": delimiter_frequency,
                    "common_patterns": [p for p, _ in self.delimiter_patterns.most_common(3)]
                }
            )
        
        # No anomalies detected
        return None
