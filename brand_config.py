"""
Simple brand configuration system.
"""

import json
import os
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

@dataclass
class BrandConfig:
    """Configuration for a specific brand."""
    brand_name: str
    field_mappings: Dict[str, str]  # field_name -> column_name
    default_data_path: Optional[str] = None
    training_data_path: Optional[str] = None
    ml_models_path: Optional[str] = None
    custom_thresholds: Optional[Dict[str, float]] = None
    enabled_fields: Optional[List[str]] = None
    
    def get_column_name(self, field_name: str) -> str:
        """Get column name for a given field, returns field_name if not mapped."""
        return self.field_mappings.get(field_name, field_name)


# Cache for loaded configurations
_brand_configs: Dict[str, BrandConfig] = {}


def load_brand_config(brand_name: str) -> BrandConfig:
    """
    Load brand configuration from JSON file.
    
    Args:
        brand_name: Name of the brand to load configuration for
        
    Returns:
        BrandConfig object
    """
    if brand_name not in _brand_configs:
        # Try brand-specific file first
        config_file = f"brand_configs/{brand_name}.json"
        
        # Fall back to single config file if brand configs directory doesn't exist
        if not os.path.exists(config_file):
            config_file = "brand_config.json"
            
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No configuration found for brand '{brand_name}'")
            
        with open(config_file, 'r') as f:
            data = json.load(f)
            
        # If using the single config file, verify it's for the requested brand
        if config_file == "brand_config.json" and data.get("brand_name") != brand_name:
            # For backward compatibility, accept any brand name with the default config
            data["brand_name"] = brand_name
            
        _brand_configs[brand_name] = BrandConfig(**data)
    
    return _brand_configs[brand_name]


def get_available_brands() -> List[str]:
    """Get list of available brand configurations."""
    brands = []
    
    # Check brand_configs directory
    if os.path.exists("brand_configs"):
        for filename in os.listdir("brand_configs"):
            if filename.endswith('.json'):
                brands.append(filename[:-5])  # Remove .json extension
                
    # Check default config
    if os.path.exists("brand_config.json"):
        with open("brand_config.json", 'r') as f:
            data = json.load(f)
            brand_name = data.get("brand_name", "default")
            if brand_name not in brands:
                brands.append(brand_name)
                
    return brands