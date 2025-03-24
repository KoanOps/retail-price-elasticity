"""
Configuration manager for the Retail Elasticity Analysis.

This module provides a centralized configuration management system with
structured configuration classes using dataclasses.
"""
import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field, fields

from utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger()

# Singleton config manager instance
_config_manager = None

def get_config() -> 'ConfigManager':
    """Get the singleton ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

@dataclass
class AppConfig:
    """Unified application configuration parameters with prefixed attributes"""
    # App settings
    results_dir: str = "results"
    create_plots: bool = True
    save_excel: bool = True
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/elasticity_analysis.log"
    
    # Model settings (with model_ prefix)
    model_type: str = "bayesian"
    model_name: str = "elasticity_model"
    model_use_seasonality: bool = True
    model_elasticity_prior_mean: float = -1.5
    model_elasticity_prior_std: float = 1.0
    model_class_effect_std: float = 0.5
    model_sku_effect_std: float = 0.25
    model_observation_error: float = 0.1
    model_n_draws: int = 1000
    model_n_tune: int = 500
    model_n_chains: int = 2
    model_target_accept: float = 0.8

    # Data settings (with data_ prefix)
    data_price_col: str = "Price_Per_Unit"
    data_quantity_col: str = "Qty_Sold"
    data_sku_col: str = "SKU_Coded"
    data_product_class_col: str = "Product_Class_Code"
    data_date_col: str = "date"
    data_log_transform_price: bool = True
    data_log_transform_quantity: bool = True
    data_path: str = "data/sales.parquet"
    data_sample_frac: float = 0.1
    data_holidays: List[str] = field(default_factory=list)
    data_events: List[str] = field(default_factory=list)
    data_column_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Mapping of legacy attribute names to prefixed names for compatibility
    _compatibility_map: Dict[str, str] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Initialize compatibility mapping after initialization"""
        # Create the compatibility map for model_config attributes
        model_fields = {
            "model_type", "model_name", "use_seasonality", "elasticity_prior_mean", 
            "elasticity_prior_std", "class_effect_std", "sku_effect_std", 
            "observation_error", "n_draws", "n_tune", "n_chains", "target_accept"
        }
        
        # Create the compatibility map for data_config attributes
        data_fields = {
            "price_col", "quantity_col", "sku_col", "product_class_col", 
            "date_col", "log_transform_price", "log_transform_quantity", 
            "data_path", "sample_frac", "holidays", "events", "column_mappings"
        }
        
        # Build the compatibility map
        self._compatibility_map = {}
        
        # Map model fields
        for field in model_fields:
            prefixed_field = f"model_{field}" if not field.startswith("model_") else field
            if hasattr(self, prefixed_field):
                self._compatibility_map[field] = prefixed_field
        
        # Map data fields
        for field in data_fields:
            prefixed_field = f"data_{field}" if not field.startswith("data_") else field
            if hasattr(self, prefixed_field):
                self._compatibility_map[field] = prefixed_field
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with fallback to default.
        
        This method supports both direct attributes and compatibility mapping.
        
        Args:
            key: Configuration key to look up
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default
    
    def __getattr__(self, name: str) -> Any:
        """
        Get attribute with support for legacy attribute names.
        
        This method is called when an attribute is not found directly,
        and it checks the compatibility mapping for legacy attribute names.
        
        Args:
            name: Attribute name to look up
            
        Returns:
            Attribute value
            
        Raises:
            AttributeError: If attribute not found in compatibility mapping
        """
        # Check if the attribute is in the compatibility map
        if name in self._compatibility_map:
            mapped_name = self._compatibility_map[name]
            return getattr(self, mapped_name)
        
        # If it's model_config or data_config, return self for compatibility
        if name in ("model_config", "data_config"):
            return self
            
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


class ConfigManager:
    """
    Unified configuration manager with typed configuration objects.
    """
    
    # Environment variable prefix for overrides
    ENV_PREFIX = "RETAIL_"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a JSON configuration file.
        """
        # Create configuration object with defaults
        self.app_config = AppConfig()
        
        # For backward compatibility
        self.model_config = self.app_config
        self.data_config = self.app_config
        self.elasticity_config = self.app_config
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
            
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self.validate()
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to a JSON configuration file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
            
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                
            # Update app config
            app_fields = {f.name for f in fields(AppConfig) if not f.name.startswith('_')}
            app_values = {k: v for k, v in config_dict.items() if k in app_fields}
            
            # Handle legacy fields with prefixing
            for key, value in config_dict.items():
                # Check if key is a known model field that needs prefixing
                model_key = f"model_{key}"
                if model_key in app_fields:
                    app_values[model_key] = value
                
                # Check if key is a known data field that needs prefixing
                data_key = f"data_{key}"
                if data_key in app_fields:
                    app_values[data_key] = value
            
            # Apply values to app_config
            for key, value in app_values.items():
                setattr(self.app_config, key, value)
                
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            
    def _apply_env_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        # Get all field names from AppConfig
        app_fields = {f.name: f for f in fields(AppConfig) if not f.name.startswith('_')}
        
        # Check for environment variables for each field
        for field_name, field_info in app_fields.items():
            env_name = f"{self.ENV_PREFIX}{field_name.upper()}"
            if env_name in os.environ:
                field_type = type(getattr(self.app_config, field_name))
                try:
                    # Convert value to appropriate type
                    if field_type == bool:
                        value = os.environ[env_name].lower() in ('true', 'yes', '1')
                    elif field_type == list:
                        value = os.environ[env_name].split(',')
                    elif field_type == dict:
                        # Parse JSON for dictionaries
                        import json
                        value = json.loads(os.environ[env_name])
                    else:
                        value = field_type(os.environ[env_name])
                    
                    # Set the value
                    setattr(self.app_config, field_name, value)
                    logger.debug(f"Applied env override for {field_name}: {value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid env value for {field_name}: {str(e)}")
            
            # Also check legacy environment variables (without prefix)
            if field_name.startswith('model_') or field_name.startswith('data_'):
                legacy_name = field_name.split('_', 1)[1]
                legacy_env = f"{self.ENV_PREFIX}{legacy_name.upper()}"
                
                if legacy_env in os.environ and legacy_env != env_name:
                    logger.warning(f"Using legacy environment variable {legacy_env}")
                    field_type = type(getattr(self.app_config, field_name))
                    try:
                        # Convert value to appropriate type as above
                        if field_type == bool:
                            value = os.environ[legacy_env].lower() in ('true', 'yes', '1')
                        elif field_type == list:
                            value = os.environ[legacy_env].split(',')
                        elif field_type == dict:
                            import json
                            value = json.loads(os.environ[legacy_env])
                        else:
                            value = field_type(os.environ[legacy_env])
                        
                        # Set the value
                        setattr(self.app_config, field_name, value)
                        logger.debug(f"Applied legacy env override for {field_name}: {value}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid legacy env value for {field_name}: {str(e)}")
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration to.
        """
        # Create dictionary from app_config
        config_dict = asdict(self.app_config)
        
        # Remove internal fields
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
                
            logger.info(f"Saved configuration to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validate the current configuration and fix common issues.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check model configuration
        if self.app_config.model_type not in ["bayesian"]:
            logger.warning(f"Unsupported model type: {self.app_config.model_type}. Using default 'bayesian'.")
            self.app_config.model_type = "bayesian"
        
        # Check data configuration
        if not self.app_config.data_path:
            logger.warning("No data path specified. Using default 'data/sales.parquet'.")
            self.app_config.data_path = "data/sales.parquet"
        
        # Check that data_path exists with friendly error
        data_path = Path(self.app_config.data_path)
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}. Please ensure it exists.")
            # Do not raise error here, just warn - allow the program to fail gracefully later
        
        # Check app configuration  
        if not self.app_config.results_dir:
            logger.warning("No results directory specified. Using default 'results'.")
            self.app_config.results_dir = "results"
            
        # Ensure minimum values for MCMC parameters
        if self.app_config.model_n_draws < 100:
            logger.warning(f"n_draws too small: {self.app_config.model_n_draws}. Setting to 100.")
            self.app_config.model_n_draws = 100
        
        if self.app_config.model_n_tune < 50:
            logger.warning(f"n_tune too small: {self.app_config.model_n_tune}. Setting to 50.")
            self.app_config.model_n_tune = 50
        
        if self.app_config.model_n_chains < 1:
            logger.warning(f"n_chains too small: {self.app_config.model_n_chains}. Setting to 1.")
            self.app_config.model_n_chains = 1
            
        return True 