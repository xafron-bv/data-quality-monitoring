from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any, Optional

from validators.validation_error import ValidationError

class ValidatorInterface(ABC):
    """
    Abstract Base Class for all Validator implementations.

    This interface guarantees that every validator provides a consistent way
    for the orchestration system to initiate a validation process on a dataset.
    """

    @abstractmethod
    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        """
        Contains the specific validation logic for a single data entry.
        This method must be implemented by subclasses.

        Args:
            value: The data from the DataFrame column to be validated.

        Returns:
            - None if the value is valid.
            - A ValidationError instance if the value is invalid.
        """
        pass

    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """
        Validates a column and returns a list of ValidationError objects.
        This method is a non-editable engine that runs the `_validate_entry` logic.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be validated.
            column_name (str): The name of the column to validate within the DataFrame.

        Returns:
            List[ValidationError]: A list of ValidationError instances representing
            the validation errors found in the specified column.
        """
        validation_errors = []
        for index, row in df.iterrows():
            data = row[column_name]

            # The implemented logic in _validate_entry is called for every row.
            validation_error = self._validate_entry(data)

            # If the custom logic returned an error, add context and add it to the list
            if validation_error:
                # Add row and column context to the validation error
                error_with_context = validation_error.with_context(
                    row_index=index,
                    column_name=column_name,
                    error_data=data
                )
                validation_errors.append(error_with_context)
                
        return validation_errors
