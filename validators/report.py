import json
import os
from typing import Any, Dict, List

import pandas as pd

from common.exceptions import ConfigurationError, FileOperationError
from validators.reporter_interface import ReporterInterface
from validators.validation_error import ValidationError


class Reporter(ReporterInterface):
    """
    Translates validation errors into human-readable messages using rule messages
    from validators/{validator_name}/rules.json.
    """

    def __init__(self, validator_name):
        self.validator_name = validator_name
        try:
            self.rule_messages = self._load_rule_messages(validator_name)
        except FileNotFoundError:
            raise FileOperationError(
                f"Rules file not found for validator '{validator_name}'",
                details={'validator_name': validator_name}
            )
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in rules file for validator '{validator_name}'",
                details={'validator_name': validator_name, 'json_error': str(e)}
            ) from e

    def _load_rule_messages(self, validator_name):
        """Load message templates from the rules.json file."""
        rules_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "validators", validator_name, "rules.json"
        )
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        # Build a map of error_type -> message (last one wins if duplicates)
        messages = {}
        for rule in rules.get('rule_flow', []):
            error_type = rule.get('error_type') or \
                         rule.get('missing_error_type') or \
                         rule.get('type_error_type') or \
                         rule.get('format_error_type') or \
                         rule.get('year_error_type') or \
                         rule.get('season_name_error_type')
            message = rule.get('message')
            if error_type and message:
                messages[error_type] = message
        return messages

    def generate_report(self, validation_errors: List[ValidationError], original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        report = []
        for error in validation_errors:
            error_code = error.error_type
            # Prefer message attached in details by rules engine; otherwise fall back to rules map
            display_message = None
            if error.details and isinstance(error.details, dict):
                display_message = error.details.get('message')
            if not display_message:
                display_message = self.rule_messages.get(error_code, "Unknown error with data: {error_data}")

            # Prepare interpolation context
            details = error.details.copy() if error.details else {}
            details['error_data'] = error.error_data
            details['probability'] = error.probability
            try:
                formatted_message = display_message.format(**details)
            except Exception:
                formatted_message = display_message

            report.append({
                "row_index": error.row_index,
                "error_data": error.error_data,
                "display_message": formatted_message,
                "column_name": error.column_name,
                "probability": error.probability,
                "error_code": error.error_type
            })
        return report
