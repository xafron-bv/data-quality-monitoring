"""
Field to column mapping for different fields in the dataset.
"""

from static_brand_config import get_field_mappings

def get_field_to_column_map():
    """
    Get field to column mapping from static configuration.
    
    Returns a dictionary mapping field names to actual column names in the data.
    """
    return get_field_mappings()
    

