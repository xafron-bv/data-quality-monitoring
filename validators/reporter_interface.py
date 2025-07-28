from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from validators.validation_error import ValidationError


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
