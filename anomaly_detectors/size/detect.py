import pandas as pd
import re
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for clothing size values.
    This detector identifies anomalies in size values by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        UNUSUAL_SIZE_VALUE = "UNUSUAL_SIZE_VALUE"
        UNUSUAL_SIZE_FORMAT = "UNUSUAL_SIZE_FORMAT"
        UNEXPECTED_NUMERIC_PATTERN = "UNEXPECTED_NUMERIC_PATTERN"
        SIZE_OUTLIER = "SIZE_OUTLIER"
    
    def __init__(self):
        """Initialize the anomaly detector with state variables."""
        self.size_frequencies = Counter()  # Track frequency of each size value
        self.numeric_sizes = []  # Store numeric sizes for statistical analysis
        self.size_formats = Counter()  # Track different format patterns
        self.total_sizes = 0
        self.size_ranges = {}  # Track the range of sizes for different categories
    
    def _extract_numeric_value(self, size: str) -> float:
        """Attempt to extract a numeric value from a size string."""
        if not isinstance(size, str):
            return None
            
        # Try to extract numeric part from the size
        match = re.search(r'(\d+\.?\d*)', size)
        if match:
            return float(match.group(1))
        return None
    
    def _get_size_format(self, size: str) -> str:
        """Generate a pattern representing the size format."""
        if not isinstance(size, str):
            return ""
            
        # Replace digits with 'D', letters with 'L', special chars kept as is
        pattern = re.sub(r'\d+', 'D', size)
        pattern = re.sub(r'[a-zA-Z]+', 'L', pattern)
        return pattern
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from size data.
        
        This method analyzes the size column to identify:
        1. Common size values
        2. Typical size formats 
        3. Normal statistical distribution of numeric sizes
        """
        # Get valid sizes
        valid_sizes = df[column_name].dropna()
        self.total_sizes = len(valid_sizes)
        
        # Extract product category information when available
        category_col = None
        for possible_col in ['article_structure_name_1', 'article_structure_name_2', 'target_area']:
            if possible_col in df.columns:
                category_col = possible_col
                break
                
        # Learn size patterns
        for idx, size in enumerate(valid_sizes):
            if not isinstance(size, str) or not size.strip():
                continue
                
            # Count the frequency of this size
            self.size_frequencies[size] += 1
            
            # Extract and store numeric value if present
            numeric_value = self._extract_numeric_value(size)
            if numeric_value is not None:
                # If category column is available, store by category
                if category_col and idx < len(df):
                    category = df.iloc[idx][category_col]
                    if category not in self.size_ranges:
                        self.size_ranges[category] = []
                    self.size_ranges[category].append(numeric_value)
                self.numeric_sizes.append(numeric_value)
            
            # Analyze format pattern
            format_pattern = self._get_size_format(size)
            self.size_formats[format_pattern] += 1
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in size values based on learned patterns.
        
        This method checks for:
        1. Unusual size values (low frequency)
        2. Unusual size formats
        3. Statistical outliers in numeric sizes
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
            
        # Check if this is an unusual size value
        size_frequency = self.size_frequencies.get(value, 0)
        if self.total_sizes > 10 and 0 < size_frequency < self.total_sizes * 0.02:  # Less than 2% frequency
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_SIZE_VALUE,
                confidence=min(0.85, 1.0 - (size_frequency / self.total_sizes)),
                details={
                    "size": value,
                    "frequency": size_frequency,
                    "total_sizes": self.total_sizes
                }
            )
        
        # Check for unusual format
        format_pattern = self._get_size_format(value)
        format_frequency = self.size_formats.get(format_pattern, 0)
        
        if self.total_sizes > 10 and 0 < format_frequency < self.total_sizes * 0.05:  # Less than 5% frequency
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_SIZE_FORMAT,
                confidence=min(0.8, 1.0 - (format_frequency / self.total_sizes)),
                details={
                    "size": value,
                    "format": format_pattern,
                    "common_formats": [f for f, _ in self.size_formats.most_common(3)]
                }
            )
        
        # Check for numeric outliers
        if self.numeric_sizes:
            numeric_value = self._extract_numeric_value(value)
            if numeric_value is not None:
                # Check if this numeric value is an outlier within its category
                if context and 'article_structure_name_1' in context:
                    category = context['article_structure_name_1']
                    if category in self.size_ranges and len(self.size_ranges[category]) >= 5:
                        category_sizes = sorted(self.size_ranges[category])
                        min_size = category_sizes[0]
                        max_size = category_sizes[-1]
                        
                        # Check if this size is well outside the normal range for this category
                        if numeric_value < min_size - (max_size - min_size) * 0.2 or numeric_value > max_size + (max_size - min_size) * 0.2:
                            return AnomalyError(
                                anomaly_type=self.ErrorCode.SIZE_OUTLIER,
                                confidence=0.85,
                                details={
                                    "size": value,
                                    "numeric_value": numeric_value,
                                    "category": category,
                                    "normal_range": f"{min_size} to {max_size}"
                                }
                            )
                
                # General check for numeric pattern
                if len(self.numeric_sizes) >= 10:
                    # Simple statistical analysis - check if outside of typical range
                    numeric_sizes = sorted(self.numeric_sizes)
                    q1_idx = int(len(numeric_sizes) * 0.25)
                    q3_idx = int(len(numeric_sizes) * 0.75)
                    q1 = numeric_sizes[q1_idx]
                    q3 = numeric_sizes[q3_idx]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    if numeric_value < lower_bound or numeric_value > upper_bound:
                        return AnomalyError(
                            anomaly_type=self.ErrorCode.UNEXPECTED_NUMERIC_PATTERN,
                            confidence=0.75,
                            details={
                                "size": value,
                                "numeric_value": numeric_value,
                                "normal_range": f"{q1} to {q3}"
                            }
                        )
        
        # No anomalies detected
        return None
