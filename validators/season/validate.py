from typing import Any, Optional

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator


class Validator(ValidatorInterface):
    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="season")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        return self._engine._validate_entry(value)
