from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union

from validators.validation_error import ValidationError

class ValidatorInterface(ABC):
    """
    Abstract Base Class for all Validator implementations.

    This interface guarantees that every validator provides a consistent way
    for the orchestration system to initiate a validation process on a dataset.
    """

    @abstractmethod
    def bulk_validate(self, df: pd.DataFrame, column_name: str) -> List[ValidationError]:
        """
        Validates a specific column in a DataFrame and returns a list of errors.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be validated.
            column_name (str): The name of the column to validate within the DataFrame.

        Returns:
            List[ValidationError]: A list of ValidationError instances representing
            the validation errors found in the specified column.
        """
        pass

class ReporterInterface(ABC):
    """
    Abstract Base Class for all Reporter implementations.

    This interface ensures that every reporter can take a structured list of
    validation errors and generate human-readable messages.
    """

    @abstractmethod
    def generate_report(self, validation_errors: List[ValidationError], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates human-readable messages for a list of validation errors.

        Args:
            validation_errors (List[ValidationError]): The list of ValidationError objects
                produced by a Validator.
            original_df (pd.DataFrame): The original DataFrame, useful for providing
                additional context in the report message.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a single reported error and must contain at least:
            - 'row_index': The integer index of the row containing the error.
            - 'error_data': The original problematic data.
            - 'display_message': A human-readable string explaining the error.
        """
        pass
