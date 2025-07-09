import pandas as pd
import re
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for product category values.
    This detector identifies anomalies in category values by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        RARE_CATEGORY = "RARE_CATEGORY"
        UNUSUAL_CATEGORY_FORMAT = "UNUSUAL_CATEGORY_FORMAT"
        INCONSISTENT_CATEGORY = "INCONSISTENT_CATEGORY"
    
    def __init__(self):
        """Initialize the anomaly detector with state variables."""
        self.category_frequencies = Counter()  # Track frequency of each category
        self.category_formats = Counter()  # Track different format patterns
        self.word_distribution = Counter()  # Count common words across categories
        self.category_relationships = defaultdict(list)  # Track category relationships with other attributes
        self.total_categories = 0
    
    def _normalize_category(self, category: str) -> str:
        """Normalize a category string for comparison."""
        if not isinstance(category, str):
            return ""
        return category.lower().strip()
    
    def _get_category_format(self, category: str) -> str:
        """Generate a pattern representing the category format."""
        if not isinstance(category, str):
            return ""
            
        # Simple format check - length and capitalization pattern
        length_group = "short" if len(category) < 8 else "medium" if len(category) < 15 else "long"
        
        has_spaces = " " in category
        space_group = "with_spaces" if has_spaces else "no_spaces"
        
        words = category.split()
        cap_pattern = ""
        for word in words:
            if word and word[0].isupper():
                cap_pattern += "U"
            else:
                cap_pattern += "l"
                
        return f"{length_group}:{space_group}:{cap_pattern}"
    
    def _extract_words(self, category: str) -> List[str]:
        """Extract individual words from a category value."""
        if not isinstance(category, str):
            return []
            
        # Split by spaces and remove any punctuation
        words = re.findall(r'[A-Za-z]+', category.lower())
        return words
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from category data.
        
        This method analyzes the category column to identify:
        1. Common category values
        2. Typical category formats
        3. Word distributions
        4. Relationships with other columns
        """
        valid_categories = df[column_name].dropna()
        self.total_categories = len(valid_categories)
        
        # Get product attributes that might be related to categories
        related_columns = []
        for col in ['article_structure_name_1', 'article_structure_name_2', 'material', 'target_area']:
            if col in df.columns and col != column_name:
                related_columns.append(col)
        
        for idx, category in enumerate(valid_categories):
            if not isinstance(category, str) or not category.strip():
                continue
                
            # Count category frequency
            norm_category = self._normalize_category(category)
            self.category_frequencies[norm_category] += 1
            
            # Analyze format pattern
            format_pattern = self._get_category_format(category)
            self.category_formats[format_pattern] += 1
            
            # Track word distribution
            words = self._extract_words(category)
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.word_distribution[word] += 1
            
            # Track relationships with other columns
            if idx < len(df) and related_columns:
                for col in related_columns:
                    related_value = df.iloc[idx][col]
                    if isinstance(related_value, str) and related_value.strip():
                        self.category_relationships[norm_category].append(
                            (col, self._normalize_category(related_value))
                        )
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in category values based on learned patterns.
        
        This method checks for:
        1. Rare categories
        2. Unusual category formats
        3. Inconsistent category relationships
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
            
        norm_category = self._normalize_category(value)
        
        # Check for rare category
        category_frequency = self.category_frequencies.get(norm_category, 0)
        if self.total_categories > 10 and 0 < category_frequency < self.total_categories * 0.03:
            return AnomalyError(
                anomaly_type=self.ErrorCode.RARE_CATEGORY,
                confidence=min(0.9, 1.0 - (category_frequency / self.total_categories)),
                details={
                    "category": value,
                    "frequency": category_frequency,
                    "total_categories": self.total_categories
                }
            )
        
        # Check for unusual format
        format_pattern = self._get_category_format(value)
        format_frequency = self.category_formats.get(format_pattern, 0)
        
        if self.total_categories > 10 and 0 < format_frequency < self.total_categories * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_CATEGORY_FORMAT,
                confidence=min(0.8, 1.0 - (format_frequency / self.total_categories)),
                details={
                    "category": value,
                    "format": format_pattern,
                    "common_formats": [f for f, _ in self.category_formats.most_common(3)]
                }
            )
        
        # Check for inconsistent category based on context
        if context and self.category_relationships.get(norm_category) and category_frequency > 0:
            # If we have learned relationships for this category and have context
            # Look for any mismatches in expected related attributes
            for related_col, related_values in self.category_relationships.items():
                # Find the most common related values for this column
                col_values = [v for c, v in related_values if c == related_col]
                if col_values:
                    col_counter = Counter(col_values)
                    common_value = col_counter.most_common(1)[0][0]
                    
                    # Check if the context has this column and if its value matches what we expect
                    if related_col in context:
                        context_value = self._normalize_category(context[related_col])
                        if context_value and context_value != common_value:
                            # The category is being used in an unusual context
                            return AnomalyError(
                                anomaly_type=self.ErrorCode.INCONSISTENT_CATEGORY,
                                confidence=0.75,
                                details={
                                    "category": value,
                                    "related_attribute": related_col,
                                    "expected_value": common_value,
                                    "actual_value": context_value
                                }
                            )
        
        # No anomalies detected
        return None
