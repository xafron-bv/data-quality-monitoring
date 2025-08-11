"""
Simple brand configuration system.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BrandConfig:
    """Configuration for a specific brand."""
    brand_name: str
    field_mappings: Dict[str, str]  # field_name -> column_name
    default_data_path: Optional[str] = None
    training_data_path: Optional[str] = None
    custom_thresholds: Optional[Dict[str, float]] = None
    enabled_fields: Optional[List[str]] = None
    field_variations: Optional[Dict[str, str]] = None  # field_name -> variation key

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
        # Get the workspace root directory (parent of common directory)
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Try brand-specific file first
        config_file = os.path.join(workspace_root, f"brand_configs/{brand_name}.json")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No configuration found for brand '{brand_name}'. Please create {config_file}")

        with open(config_file, 'r') as f:
            data = json.load(f)

        _brand_configs[brand_name] = BrandConfig(**data)

    return _brand_configs[brand_name]


def get_available_brands() -> List[str]:
    """Get list of available brand configurations."""
    brands = []

    # Get the workspace root directory (parent of common directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check brand_configs directory
    brand_configs_dir = os.path.join(workspace_root, "brand_configs")
    if os.path.exists(brand_configs_dir):
        for filename in os.listdir(brand_configs_dir):
            if filename.endswith('.json'):
                brands.append(filename[:-5])  # Remove .json extension

    return brands
