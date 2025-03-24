#!/usr/bin/env python3
"""
Default configuration for SKU Elasticity Analysis.
This module contains default values for various model parameters and settings.
"""
import os
from typing import Dict, Any, Optional

# Define the base directory for the project (2 levels up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data processing defaults
DEFAULT_DATA_PATH = "Retail/data/sales.parquet"
DEFAULT_SAMPLE_FRAC = 0.05
DEFAULT_MIN_OBSERVATIONS = 5
DEFAULT_MAX_PRODUCT_CLASSES = 10
DEFAULT_TRAIN_END_DATE = "2023-10-15"

# Results and output defaults
DEFAULT_RESULTS_DIR = "Retail/results"
DEFAULT_LOG_FILE = os.path.join(DEFAULT_RESULTS_DIR, "elasticity_analysis.log")
DEFAULT_LOG_LEVEL = "INFO"

# Model defaults
DEFAULT_MODEL_TYPE = "basic_bayesian"  # Options: "basic_bayesian", "advanced_bayesian"
DEFAULT_SAMPLING_PARAMS = {
    "draws": 1000,
    "tune": 2000,
    "chains": 3,
    "cores": 1,
    "target_accept": 0.95,
    "return_inferencedata": True,
    "random_seed": 42
}

# Basic Bayesian Model priors
BASIC_MODEL_PRIORS = {
    "mu_alpha": {
        "mu": 0.0,
        "sigma": 10.0
    },
    "sigma_alpha": {
        "nu": 3.0,
        "sigma": 2.0
    },
    "mu_beta": {
        "mu": -1.0,  # Prior expectation: negative elasticity
        "sigma": 1.0
    },
    "sigma_beta": {
        "nu": 3.0,
        "sigma": 0.5
    }
}
# Visualization settings
VISUALIZATION_SETTINGS = {
    "plot_dpi": 120,
    "figsize_medium": (10, 6),
    "figsize_large": (14, 8),
    "style": "seaborn-v0_8-whitegrid",
    "palette": "viridis",
    "save_format": "png"
}

# Model diagnostics settings
DIAGNOSTICS_SETTINGS = {
    "bins": 30,
    "credible_interval": 0.89,
    "include_hdi": True,
    "max_samples_plot": 5000
}

# Default configuration dictionary
DEFAULT_CONFIG = {
    "directories": {
        "data_dir": os.path.join(BASE_DIR, "data"),
        "results_dir": os.path.join(BASE_DIR, "results", "sku_elasticity"),
        "diagnostics_dir": os.path.join(BASE_DIR, "results", "sku_elasticity", "diagnostics"),
        "models_dir": os.path.join(BASE_DIR, "results", "sku_elasticity", "models"),
        "logs_dir": os.path.join(BASE_DIR, "logs"),
    },
    "data_processing": {
        "default_sample_frac": DEFAULT_SAMPLE_FRAC,
        "min_observations_per_sku": DEFAULT_MIN_OBSERVATIONS,
        "test_set_start_date": DEFAULT_TRAIN_END_DATE,
        "price_col": "Price_Per_Unit",
        "quantity_col": "Qty_Sold",
        "date_col": "Sold_Date",
        "sku_col": "SKU_Coded",
        "store_col": "Store_Number",
        "product_class_col": "Product_Class_Code",
        "generate_diagnostics": True,
    },
    "model": {
        "standard": {
            "n_draws": 1000,
            "tune": 1000,
            "chains": 2,
            "target_accept": 0.95,
            "return_inferencedata": True,
            "random_seed": 42,
        },
        "advanced": {
            "n_draws": 1000,
            "tune": 1000,
            "chains": 2,
            "target_accept": 0.95,
            "return_inferencedata": True,
            "random_seed": 42,
        },
    },
    "visualization": {
        "default_figsize": (10, 6),
        "dpi": 100,
        "save_format": "png",
        "style": "seaborn-v0_8-whitegrid",
    },
}

def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Optional section name to retrieve only a subset of the config
        
    Returns:
        Dictionary containing configuration settings
    """
    if section is not None:
        if section in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[section]
        else:
            raise ValueError(f"Config section '{section}' not found")
    return DEFAULT_CONFIG

def ensure_directories_exist() -> None:
    """
    Ensure that all directories specified in the configuration exist.
    Creates them if they don't exist.
    """
    # Import here to avoid circular imports
    from utils.common import ensure_dir_exists
    
    for _, directory in DEFAULT_CONFIG["directories"].items():
        ensure_dir_exists(directory)

# Create necessary directories when module is imported
ensure_directories_exist()

if __name__ == "__main__":
    # Print configuration when run directly
    import json
    print("Default Configuration:")
    print(json.dumps(DEFAULT_CONFIG, indent=2))
    
    # Test directory creation
    print("\nVerifying Directories:")
    for name, path in DEFAULT_CONFIG["directories"].items():
        exists = os.path.exists(path)
        print(f"{name}: {path} {'✓' if exists else '✗'}") 