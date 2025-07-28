"""
Field to column mapping for different fields in the dataset.
"""

from brand_config import load_brand_config


def get_field_to_column_map(brand_name: str = "esqualo"):
    """
    Get field to column mapping for a specific brand.

    Args:
        brand_name: Name of the brand (defaults to "esqualo" for backward compatibility)

    Returns a dictionary mapping field names to actual column names in the data.
    """
    config = load_brand_config(brand_name)
    return config.field_mappings
