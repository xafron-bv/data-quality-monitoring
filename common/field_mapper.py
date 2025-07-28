"""
Field mapper interface for translating between standard field names and brand-specific column names.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FieldMapper:
    """Maps standard field names to brand-specific column names."""

    def __init__(self, field_mappings: Dict[str, str], brand_name: Optional[str] = None):
        """
        Initialize the field mapper.

        Args:
            field_mappings: Dictionary mapping standard field names to column names
            brand_name: Optional brand name for identification
        """
        self.field_mappings = field_mappings
        self._brand_name = brand_name
        # Create reverse mapping for column to field lookup
        self.column_to_field = {v: k for k, v in field_mappings.items()}

    def get_column_name(self, field_name: str) -> Optional[str]:
        """Get the column name for a given field name."""
        return self.field_mappings.get(field_name)

    def get_field_name(self, column_name: str) -> Optional[str]:
        """Get the field name for a given column name."""
        return self.column_to_field.get(column_name)

    def list_fields(self) -> List[str]:
        """List all available field names."""
        return list(self.field_mappings.keys())

    def list_columns(self) -> List[str]:
        """List all mapped column names."""
        return list(self.field_mappings.values())

    def get_brand_name(self) -> Optional[str]:
        """Get the brand name associated with this mapper."""
        return self._brand_name

    def get_available_fields(self) -> List[str]:
        """Get all available field names. Alias for list_fields()."""
        return self.list_fields()

    def validate_column_exists(self, df, field_name: str) -> Optional[str]:
        """Validate that a column exists in the dataframe for the given field."""
        column_name = self.get_column_name(field_name)
        if column_name and column_name in df.columns:
            return column_name
        return None

    @classmethod
    def from_brand(cls, brand_name: str) -> 'FieldMapper':
        """Get field mapper for a specific brand."""
        from common.brand_config import load_brand_config
        config = load_brand_config(brand_name)
        return cls(config.field_mappings, brand_name)
