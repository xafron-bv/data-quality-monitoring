"""Exception hierarchy for data quality monitoring system."""

from typing import Dict, Any, Optional


class DataQualityError(Exception):
    """Base exception for all data quality monitoring operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(DataQualityError):
    """Configuration-related errors."""
    pass


class ValidationError(DataQualityError):
    """Data validation failures."""
    pass


class DetectionError(DataQualityError):
    """Detection operation failures."""
    pass


class ModelError(DetectionError):
    """ML model operation failures."""
    pass


class DataError(DataQualityError):
    """Data-related errors."""
    pass


class FileOperationError(DataQualityError):
    """File operation failures."""
    pass 