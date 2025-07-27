"""
Simple static brand configuration loader.
"""

import json
import os
from typing import Dict, Optional, List, Any

# Global variable to cache the loaded configuration
_brand_config: Optional[Dict[str, Any]] = None

def load_brand_config(config_file: str = "brand_config.json") -> Dict[str, Any]:
    """Load brand configuration from a static JSON file."""
    global _brand_config
    if _brand_config is None:
        with open(config_file, 'r') as f:
            _brand_config = json.load(f)
    return _brand_config

def get_field_mappings() -> Dict[str, str]:
    """Get field mappings from the configuration."""
    config = load_brand_config()
    return config.get("field_mappings", {})

def get_brand_name() -> str:
    """Get the brand name from configuration."""
    config = load_brand_config()
    return config.get("brand_name", "default")

def get_data_path(path_type: str = "default") -> Optional[str]:
    """Get data path from configuration."""
    config = load_brand_config()
    if path_type == "training":
        return config.get("training_data_path")
    else:
        return config.get("default_data_path")

def get_ml_models_path() -> Optional[str]:
    """Get ML models path from configuration."""
    config = load_brand_config()
    return config.get("ml_models_path")

def get_enabled_fields() -> List[str]:
    """Get enabled fields from configuration."""
    config = load_brand_config()
    return config.get("enabled_fields", [])

def get_custom_thresholds() -> Dict[str, float]:
    """Get custom thresholds from configuration."""
    config = load_brand_config()
    return config.get("custom_thresholds", {})

def get_column_name(field_name: str) -> str:
    """Get column name for a given field, returns field_name if not mapped."""
    mappings = get_field_mappings()
    return mappings.get(field_name, field_name)