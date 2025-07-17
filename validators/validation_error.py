from enum import Enum
from typing import Dict, Any, Optional, Union

class ValidationError:
    """
    A standardized error class for validation errors.
    
    This class enforces that all validation errors include:
    - An error code (from ErrorCode enum)
    - A probability level (float between 0 and 1)
    - Optional details as a dictionary
    """
    
    def __init__(self, error_type: Union[str, Enum], probability: float, details: Optional[Dict[str, Any]] = None, 
                row_index: Optional[int] = None, column_name: Optional[str] = None, error_data: Any = None):
        """
        Initialize a ValidationError with required fields.
        
        Args:
            error_type: The error code/type (from ErrorCode enum)
            probability: A value between 0 and 1 indicating the probability that this is an error
            details: Optional dictionary containing additional error details
            row_index: Optional row index where the error occurred
            column_name: Optional column name where the error occurred
            error_data: Optional original data that caused the error
        
        Raises:
            ValueError: If probability is not between 0 and 1
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        
        self.error_type = error_type
        self.probability = probability
        self.details = details or {}
        self.row_index = row_index
        self.column_name = column_name
        self.error_data = error_data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary format compatible with the validator interface.
        
        Returns:
            Dict containing the error information
        """
        result = {
            "error_code": self.error_type,
            "probability": self.probability,
            "details": self.details
        }
        
        if self.row_index is not None:
            result["row_index"] = self.row_index
            
        if self.column_name is not None:
            result["column_name"] = self.column_name
            
        if self.error_data is not None:
            result["error_data"] = self.error_data
            
        return result
    
    def with_context(self, row_index: int, column_name: str, error_data: Any) -> 'ValidationError':
        """
        Creates a new ValidationError with the same error information but with added context
        
        Args:
            row_index: The index of the row where the error was found
            column_name: The name of the column where the error was found
            error_data: The original data that caused the error
            
        Returns:
            A new ValidationError instance with the added context
        """
        return ValidationError(
            error_type=self.error_type,
            probability=self.probability,
            details=self.details,
            row_index=row_index,
            column_name=column_name,
            error_data=error_data
        )
