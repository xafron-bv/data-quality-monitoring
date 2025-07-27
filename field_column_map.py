"""
Field to column mapping for different fields in the dataset.
"""

from brand_configs import get_brand_config_manager

def get_field_to_column_map():
    """
    Get field to column mapping based on current brand configuration.
    
    Returns a dictionary mapping field names to actual column names in the data.
    """
    brand_manager = get_brand_config_manager()
    current_brand = brand_manager.get_current_brand()
    
    if current_brand and hasattr(current_brand, 'field_mappings'):
        return current_brand.field_mappings
    else:
        raise ValueError("No brand configured or field mappings not found.")
    

