def get_field_to_column_map(brand_name=None):
    """
    Get the mapping from field names to column names for a specific brand.
    
    Args:
        brand_name: Name of the brand. If None, uses current brand.
    
    Returns:
        Dictionary mapping field names to column names.
    
    Raises:
        ValueError: If no brand is configured and none specified.
    """
    from brand_configs import get_brand_config_manager
    
    manager = get_brand_config_manager()
    
    # If no brand specified, try to get current brand
    if brand_name is None:
        config = manager.get_current_brand()
        if config:
            return config.field_mappings
        else:
            raise ValueError("No brand configured. Please specify a brand or set current brand.")
    else:
        config = manager.get_brand(brand_name)
        if config:
            return config.field_mappings
        else:
            raise ValueError(f"Brand '{brand_name}' not found in configurations.")
    

