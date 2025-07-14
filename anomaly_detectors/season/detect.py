import pandas as pd
import re
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
from datetime import datetime

from anomaly_detectors.anomaly_detector_interface import AnomalyDetectorInterface
from anomaly_detectors.anomaly_error import AnomalyError


class AnomalyDetector(AnomalyDetectorInterface):
    """
    An anomaly detector for season values (e.g., "2022 Fall", "2023 Summer").
    This detector identifies anomalies in season descriptions by learning patterns
    from the data and flagging instances that deviate from those patterns.
    """
    class ErrorCode(str, Enum):
        """Enumeration for anomaly detector error codes."""
        UNCOMMON_SEASON_FORMAT = "UNCOMMON_SEASON_FORMAT"
        UNUSUAL_SEASON = "UNUSUAL_SEASON"
        TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY" 
        INCONSISTENT_SEASON_PATTERN = "INCONSISTENT_SEASON_PATTERN"
    
    def __init__(self):
        """Initialize the anomaly detector with state variables."""
        self.season_frequencies = Counter()  # Track frequency of each season
        self.season_formats = Counter()  # Track different season formats
        self.year_distribution = Counter()  # Track year frequencies
        self.season_name_distribution = Counter()  # Track season name frequencies
        self.total_seasons = 0
        self.current_year = datetime.now().year
    
    def _extract_year_and_season(self, season: str) -> tuple:
        """Extract year and season name from a season string."""
        if not isinstance(season, str):
            return None, None
            
        # Common pattern: "2022 Fall" or similar
        match = re.match(r'(\d{4})\s+(.+)', season)
        if match:
            year = int(match.group(1))
            season_name = match.group(2)
            return year, season_name
            
        # Try to extract any year
        year_match = re.search(r'\d{4}', season)
        if year_match:
            year = int(year_match.group(0))
            # Remove year to get season
            season_name = season.replace(year_match.group(0), '').strip()
            return year, season_name
            
        return None, None
    
    def _get_season_format(self, season: str) -> str:
        """Generate a pattern representing the season format."""
        if not isinstance(season, str):
            return ""
            
        # Replace digits with 'D', letters with 'L', space with 'S'
        pattern = re.sub(r'\d+', 'D', season)
        pattern = re.sub(r'[a-zA-Z]+', 'L', pattern)
        pattern = re.sub(r'\s+', 'S', pattern)
        return pattern
    
    def learn_patterns(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Learn normal patterns from season data.
        
        This method analyzes the season column to identify:
        1. Common season values
        2. Typical season formats
        3. Year and season name distributions
        """
        valid_seasons = df[column_name].dropna()
        self.total_seasons = len(valid_seasons)
        
        for season in valid_seasons:
            if not isinstance(season, str) or not season.strip():
                continue
                
            # Count season frequency
            self.season_frequencies[season] += 1
            
            # Extract and analyze year and season components
            year, season_name = self._extract_year_and_season(season)
            if year:
                self.year_distribution[year] += 1
            if season_name:
                self.season_name_distribution[season_name.lower()] += 1
            
            # Analyze format pattern
            format_pattern = self._get_season_format(season)
            self.season_formats[format_pattern] += 1
    
    def _detect_anomaly(self, value: Any, context: Dict[str, Any] = None) -> Optional[AnomalyError]:
        """
        Detect anomalies in season values based on learned patterns.
        
        This method checks for:
        1. Uncommon season formats
        2. Unusual seasons (low frequency)
        3. Temporal anomalies (unusually old or future seasons)
        4. Inconsistent season patterns
        """
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            return None
        
        # Check for unusual season
        season_frequency = self.season_frequencies.get(value, 0)
        if self.total_seasons > 10 and 0 < season_frequency < self.total_seasons * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNUSUAL_SEASON,
                anomaly_score=min(0.85, 1.0 - (season_frequency / self.total_seasons)),
                details={
                    "season": value,
                    "frequency": season_frequency,
                    "total_seasons": self.total_seasons
                }
            )
        
        # Check for uncommon format
        format_pattern = self._get_season_format(value)
        format_frequency = self.season_formats.get(format_pattern, 0)
        
        if self.total_seasons > 10 and 0 < format_frequency < self.total_seasons * 0.05:
            return AnomalyError(
                anomaly_type=self.ErrorCode.UNCOMMON_SEASON_FORMAT,
                anomaly_score=min(0.8, 1.0 - (format_frequency / self.total_seasons)),
                details={
                    "season": value,
                    "format": format_pattern,
                    "common_formats": [f for f, _ in self.season_formats.most_common(3)]
                }
            )
        
        # Extract year and season components
        year, season_name = self._extract_year_and_season(value)
        
        # Check for temporal anomalies
        if year:
            # Check for unusually old or future years
            years = sorted(self.year_distribution.keys())
            if years:
                min_year = min(years)
                max_year = max(years)
                typical_range = max_year - min_year
                
                # Year is too far in the past or too far in the future
                if (year < min_year - 1) or (year > self.current_year + 2):
                    return AnomalyError(
                        anomaly_type=self.ErrorCode.TEMPORAL_ANOMALY,
                        anomaly_score=0.85,
                        details={
                            "season": value,
                            "year": year,
                            "typical_range": f"{min_year} to {max_year}",
                            "current_year": self.current_year
                        }
                    )
        
        # Check for inconsistent season pattern
        if season_name:
            # Check if this season name is unusual
            season_name_lower = season_name.lower()
            name_frequency = self.season_name_distribution.get(season_name_lower, 0)
            
            if self.total_seasons > 10 and 0 < name_frequency < self.total_seasons * 0.05:
                # This is an unusual season name
                return AnomalyError(
                    anomaly_type=self.ErrorCode.INCONSISTENT_SEASON_PATTERN,
                    anomaly_score=min(0.75, 1.0 - (name_frequency / self.total_seasons)),
                    details={
                        "season": value,
                        "season_name": season_name,
                        "common_season_names": [s for s, _ in self.season_name_distribution.most_common(4)]
                    }
                )
        
        # No anomalies detected
        return None
