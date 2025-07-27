"""
Brand configuration management system.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from field_mapper import FieldMapper


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrandConfig':
        """Create BrandConfig from dictionary."""
        # Remove any comment fields
        cleaned_data = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**cleaned_data)


class BrandConfigManager:
    """Manages brand configurations for the data quality monitoring system."""
    
    # Default configuration directory - use absolute path relative to this file
    DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand_configs")
    
    def __init__(self, config_dir: Optional[str] = None, specific_brand_file: Optional[str] = None):
        """
        Initialize the brand config manager.
        
        Args:
            config_dir: Directory containing brand JSON files. Defaults to 'brand_configs'.
            specific_brand_file: Path to a specific brand JSON file. If provided, only loads this file.
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.specific_brand_file = specific_brand_file
        self._configs: Dict[str, BrandConfig] = {}
        self._current_brand: Optional[str] = None
        
        # Load configurations
        if specific_brand_file and os.path.exists(specific_brand_file):
            self._load_from_file(specific_brand_file)
        else:
            self._load_from_directory()
    
    def _load_from_directory(self):
        """Load all brand configurations from the config directory."""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            print(f"Created brand configs directory: {self.config_dir}")
            print(f"Please add brand configuration JSON files to this directory.")
            return
        
        # Load all JSON files in the directory
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json') and not filename.startswith('example_'):
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config_data = json.load(f)
                    
                    # Skip template files
                    if config_data.get('brand_name', '').startswith('YOUR_'):
                        continue
                    
                    brand_name = config_data.get('brand_name', os.path.splitext(filename)[0])
                    self._configs[brand_name] = BrandConfig.from_dict(config_data)
                    print(f"Loaded brand configuration: {brand_name}")
                except Exception as e:
                    print(f"Error loading brand configuration from {filepath}: {e}")
        
        if not self._configs:
            print(f"Warning: No brand configurations found in {self.config_dir}")
            print("Please create brand configuration JSON files based on example_brand_template.json")
    
    def _load_from_file(self, filepath: str):
        """Load brand configuration from a single JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            brand_name = config_data.get('brand_name')
            if not brand_name:
                brand_name = os.path.splitext(os.path.basename(filepath))[0]
                config_data['brand_name'] = brand_name
            
            self._configs[brand_name] = BrandConfig.from_dict(config_data)
            print(f"Loaded brand configuration: {brand_name} from {filepath}")
        except Exception as e:
            print(f"Error loading brand configuration from {filepath}: {e}")
    
    def save_brand(self, brand_config: BrandConfig, filepath: Optional[str] = None):
        """Save a single brand configuration to JSON file."""
        if not filepath:
            filepath = os.path.join(self.config_dir, f"{brand_config.brand_name}.json")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(brand_config.to_dict(), f, indent=2)
        
        print(f"Saved brand configuration to: {filepath}")
    
    def add_brand(self, config: BrandConfig):
        """Add or update a brand configuration."""
        self._configs[config.brand_name] = config
    
    def get_brand(self, brand_name: str) -> Optional[BrandConfig]:
        """Get configuration for a specific brand."""
        return self._configs.get(brand_name)
    
    def set_current_brand(self, brand_name: str):
        """Set the current active brand."""
        if brand_name not in self._configs:
            raise ValueError(f"Unknown brand: {brand_name}. Available brands: {list(self._configs.keys())}")
        self._current_brand = brand_name
    
    def get_current_brand(self) -> Optional[BrandConfig]:
        """Get the current active brand configuration."""
        if not self._current_brand:
            return None
        return self._configs.get(self._current_brand)
    
    def list_brands(self) -> List[str]:
        """List all available brand names."""
        return list(self._configs.keys())
    
    def get_field_mapper(self, brand_name: Optional[str] = None) -> 'FieldMapper':
        """
        Get a FieldMapper instance for the specified brand.
        
        Args:
            brand_name: Brand to get mapper for. Uses current brand if not specified.
            
        Returns:
            FieldMapper instance configured for the brand.
        """
        brand_name = brand_name or self._current_brand
        if not brand_name:
            raise ValueError("No brand specified and no current brand set")
        
        config = self.get_brand(brand_name)
        if not config:
            raise ValueError(f"No configuration found for brand: {brand_name}")
        
        return FieldMapper(config.field_mappings)
    
    def get_data_path(self, brand_name: Optional[str] = None, 
                      path_type: str = "default") -> Optional[str]:
        """
        Get data path for a brand.
        
        Args:
            brand_name: Brand name. Uses current brand if not specified.
            path_type: Type of path - "default" or "training"
            
        Returns:
            Path to data file or None if not configured.
        """
        brand_name = brand_name or self._current_brand
        if not brand_name:
            return None
        
        config = self.get_brand(brand_name)
        if not config:
            return None
        
        if path_type == "training":
            return config.training_data_path
        else:
            return config.default_data_path


# Global instance for easy access
_brand_config_manager = None


def get_brand_config_manager(config_dir: Optional[str] = None, 
                           specific_brand_file: Optional[str] = None) -> BrandConfigManager:
    """
    Get or create the global brand config manager instance.
    
    Args:
        config_dir: Directory containing brand JSON files.
        specific_brand_file: Path to a specific brand JSON file.
    
    Returns:
        BrandConfigManager instance.
    """
    global _brand_config_manager
    if _brand_config_manager is None:
        _brand_config_manager = BrandConfigManager(config_dir, specific_brand_file)
    return _brand_config_manager


def create_brand_config_template(brand_name: str, output_file: Optional[str] = None):
    """
    Create a template configuration file for a new brand.
    
    Note: Validation and anomaly detection rules are global and shared across all brands.
    Only column mappings and data paths are brand-specific.
    
    Args:
        brand_name: Name of the new brand
        output_file: Path to save the template. If not provided, saves to brand_configs/{brand_name}.json
    """
    if not output_file:
        output_file = os.path.join("brand_configs", f"{brand_name}.json")
    
    template = {
        "brand_name": brand_name,
        "field_mappings": {
            "category": "YOUR_CATEGORY_COLUMN",
            "color_name": "YOUR_COLOR_COLUMN",
            "ean": "YOUR_EAN_COLUMN",
            "article_number": "YOUR_ARTICLE_NUMBER_COLUMN",
            "colour_code": "YOUR_COLOR_CODE_COLUMN",
            "customs_tariff_number": "YOUR_CUSTOMS_TARIFF_COLUMN",
            "description_short_1": "YOUR_SHORT_DESCRIPTION_COLUMN",
            "long_description_nl": "YOUR_LONG_DESCRIPTION_COLUMN",
            "material": "YOUR_MATERIAL_COLUMN",
            "product_name_en": "YOUR_PRODUCT_NAME_COLUMN",
            "size": "YOUR_SIZE_COLUMN",
            "care_instructions": "YOUR_CARE_INSTRUCTIONS_COLUMN",
            "season": "YOUR_SEASON_COLUMN",
            "manufactured_in": "YOUR_MANUFACTURED_IN_COLUMN",
            "supplier": "YOUR_SUPPLIER_COLUMN",
            "brand": "YOUR_BRAND_COLUMN",
            "collection": "YOUR_COLLECTION_COLUMN"
        },
        "default_data_path": f"data/{brand_name}_data.csv",
        "training_data_path": f"data/{brand_name}_training.csv",
        "ml_models_path": f"anomaly_detectors/ml_based/results/{brand_name}",
        "enabled_fields": ["material", "color_name", "category", "size"],
        "custom_thresholds": {
            "validation": 0.0,
            "anomaly": 0.7,
            "ml": 0.7,
            "llm": 0.6
        },
        "_comment": "Validation and anomaly detection rules are global and shared across all brands."
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Brand configuration template created at: {output_file}")
    print(f"Edit this file to set the correct column mappings for {brand_name}")
    print(f"Note: Validation and anomaly detection rules are global and work across all brands.")