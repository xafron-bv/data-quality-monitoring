import re
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from validators.validation_error import ValidationError
from validators.validator_interface import ValidatorInterface
from validators.rules_engine import JsonRulesValidator


class Validator(ValidatorInterface):
    """
    Validator for clothing size values, now delegated to JSON rules.
    """

    def __init__(self) -> None:
        self._engine = JsonRulesValidator(field_name="size")

    def _validate_entry(self, value: Any) -> Optional[ValidationError]:
        return self._engine._validate_entry(value)
