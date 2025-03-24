"""
Constants for the retail price elasticity analysis models.

This module centralizes all constants and default configuration values used 
throughout the codebase to eliminate magic numbers and improve configuration
management.
"""
from typing import Dict, Any

# =======================================================
# Default Model Configuration Constants
# =======================================================

# Elasticity prior distribution parameters
DEFAULT_ELASTICITY_PRIOR_MEAN = -1.0
DEFAULT_ELASTICITY_PRIOR_STD = 0.5
DEFAULT_CLASS_EFFECT_STD = 0.25
DEFAULT_SKU_EFFECT_STD = 0.25
DEFAULT_SEASONAL_EFFECT_STD = 0.1
DEFAULT_SIGMA_PRIOR = 0.5

# MCMC sampling parameters
DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 4
DEFAULT_TARGET_ACCEPT = 0.95

# Data processing parameters
DEFAULT_INVALID_PRICE_FACTOR = 10.0  # divisor for minimum price when replacing non-positive prices
DEFAULT_INVALID_QUANTITY_FACTOR = 10.0  # divisor for minimum quantity when replacing non-positive quantities

# Visualization parameters
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 100
DEFAULT_PLOT_TYPE = 'scatter'

# Time-related constants
MONTHS_IN_YEAR = 12

# =======================================================
# Elasticity Categories
# =======================================================

# Elasticity classification thresholds
ELASTIC_THRESHOLD = -1.0  # threshold for elastic products (e < -1)
POSITIVE_ELASTICITY_THRESHOLD = 0.0  # threshold for abnormal positive elasticity 

# =======================================================
# Default Configuration Values
# =======================================================

# Default model configuration
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    # Prior distribution parameters
    "elasticity_prior_mean": DEFAULT_ELASTICITY_PRIOR_MEAN,
    "elasticity_prior_std": DEFAULT_ELASTICITY_PRIOR_STD,
    "class_effect_std": DEFAULT_CLASS_EFFECT_STD,
    "sku_effect_std": DEFAULT_SKU_EFFECT_STD,
    "seasonal_effect_std": DEFAULT_SEASONAL_EFFECT_STD,
    
    # MCMC parameters
    "n_draws": DEFAULT_DRAWS,
    "n_tune": DEFAULT_TUNE,
    "n_chains": DEFAULT_CHAINS,
    "target_accept": DEFAULT_TARGET_ACCEPT,
}

# Default data configuration
DEFAULT_DATA_CONFIG: Dict[str, Any] = {
    "sku_col": "sku",
    "product_class_col": "product_class",
    "price_col": "price",
    "quantity_col": "quantity",
    "date_col": "date",
    "log_transform_price": True,
    "log_transform_quantity": True,
    "include_seasonality": True,
}

# Metrics for missing values in error handling
DEFAULT_ERROR_VALUE = 1.0  # Used to replace NaN or infinity in error calculations 