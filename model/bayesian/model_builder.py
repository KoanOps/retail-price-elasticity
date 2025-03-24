"""
Bayesian Model Builder for Price Elasticity Estimation.

This module creates and configures hierarchical Bayesian models for estimating
price elasticity at both individual SKU and product class levels.
"""

import numpy as np
import pandas as pd
import logging
import pymc as pm
from typing import Dict, List, Optional, Any, Tuple, Union
import os
import time
from dataclasses import dataclass

from model.exceptions import ModelBuildError
from model.constants import (
    DEFAULT_ELASTICITY_PRIOR_MEAN,
    DEFAULT_ELASTICITY_PRIOR_STD,
    DEFAULT_CLASS_EFFECT_STD,
    DEFAULT_SKU_EFFECT_STD,
    DEFAULT_SEASONAL_EFFECT_STD,
    DEFAULT_SIGMA_PRIOR,
    DEFAULT_MODEL_CONFIG
)

# Type imports for the PyMC models (for development/type checking only)
try:
    import pymc as _pm_type_hint  # For type hints only
except ImportError:
    pass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelData:
    """
    Container for prepared model data.
    
    ASSUMPTIONS:
    - X contains log-transformed price data
    - y contains log-transformed quantity data
    - Each SKU is mapped to exactly one product class
    - Indices are 0-based and contiguous
    - All arrays have appropriate dimensions and compatible indices
    
    EDGE CASES:
    - Empty data arrays (n=0) will cause model failure
    - Missing product classes for any SKU will cause index errors
    - Non-numeric data in X or y will cause PyMC errors during sampling
    """
    X: np.ndarray  # Price data (log-transformed)
    y: np.ndarray  # Quantity data (log-transformed)
    sku_idx: np.ndarray  # SKU indices for each observation
    product_class_idx: np.ndarray  # Product class indices for each SKU
    n_skus: int  # Number of unique SKUs
    n_classes: int  # Number of unique product classes
    seasonality_features: Optional[np.ndarray] = None  # Seasonal dummy variables
    sku_names: Optional[np.ndarray] = None  # Original SKU identifiers
    class_names: Optional[np.ndarray] = None  # Original product class identifiers


class BayesianModelBuilder:
    """
    Builds PyMC model graphs for Bayesian elasticity estimation.
    
    ASSUMPTIONS:
    - Log-log model is appropriate (log(quantity) ~ log(price))
    - Hierarchical structure captures true elasticity distribution
    - Normal distribution is appropriate for all elasticity components
    - Partial pooling improves inference over complete pooling or no pooling
    - Error term is normally distributed and homoscedastic
    - PyMC package is available and properly configured
    
    EDGE CASES:
    - Very small datasets may lead to poor convergence
    - High collinearity in seasonality features may cause identification issues
    - Extreme priors may lead to sampling difficulties
    - Models with many parameters may require significant computational resources
    - Highly skewed data may violate normality assumptions
    
    MODELING CHOICES:
    - Global mean elasticity (hyperprior)
    - Class-level partial pooling
    - SKU-level partial pooling
    - Optional seasonality effects
    - Base demand parameter
    
    Responsibilities:
    - Constructing the PyMC model graph
    - Setting prior distributions
    - Defining likelihood functions
    - Configuring model parameters
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        use_seasonality: bool = True,
        pymc=None  # PyMC module injected at runtime
    ):
        """
        Initialize the model builder.
        
        Args:
            model_config: Configuration dictionary for the model
            use_seasonality: Whether to include seasonality in the model
            pymc: PyMC module injected at runtime
        """
        self.model_config = model_config or DEFAULT_MODEL_CONFIG.copy()
        self.use_seasonality = use_seasonality
        self.pymc = pymc
        
        # Get model parameters from config with defaults from constants
        self.elasticity_prior_mean = self.model_config.get("elasticity_prior_mean", DEFAULT_ELASTICITY_PRIOR_MEAN)
        self.elasticity_prior_std = self.model_config.get("elasticity_prior_std", DEFAULT_ELASTICITY_PRIOR_STD)
        self.class_effect_std = self.model_config.get("class_effect_std", DEFAULT_CLASS_EFFECT_STD)
        self.sku_effect_std = self.model_config.get("sku_effect_std", DEFAULT_SKU_EFFECT_STD)
        self.seasonal_effect_std = self.model_config.get("seasonal_effect_std", DEFAULT_SEASONAL_EFFECT_STD)
        
        # Initialize model
        self._model = None
        
    def build_model(self, data: pd.DataFrame) -> pm.Model:
        """
        Build a Bayesian hierarchical model for price elasticity estimation.
        
        Args:
            data: Prepared data with all required columns
            
        Returns:
            PyMC model for elasticity estimation
        """
        try:
            # First, prepare the model data using the existing method
            model_data = self._prepare_model_data(data)
            
            # Extract data from model_data object
            X = model_data.X  # Log price data
            y = model_data.y  # Log quantity data
            sku_idx = model_data.sku_idx  # SKU indices
            product_class_idx = model_data.product_class_idx  # Product class indices for each SKU
            n_skus = model_data.n_skus
            n_product_classes = len(np.unique(product_class_idx))
            
            logger.info(f"Building model with {len(X)} observations, "
                       f"{n_product_classes} product classes, {n_skus} SKUs")
            
            # Check for high collinearity in seasonality features
            seasonal_X = model_data.seasonality_features
            
            # Create PyMC model
            with pm.Model() as model:
                # Prior for global mean elasticity
                elasticity_mu = pm.Normal(
                    'elasticity_mu',
                    mu=self.elasticity_prior_mean,
                    sigma=self.elasticity_prior_std,
                    initval=self.elasticity_prior_mean
                )
                
                # Hierarchical priors for product class effects (deviations from global mean)
                elasticity_class_effect = pm.Normal(
                    'elasticity_class_effect',
                    mu=0,
                    sigma=self.class_effect_std,
                    shape=n_product_classes
                )
                
                # Hierarchical priors for SKU effects (deviations from product class means)
                elasticity_sku_raw = pm.Normal(
                    'elasticity_sku_raw',
                    mu=0,
                    sigma=1,
                    shape=n_skus
                )
                
                # Non-centered parameterization for better sampling
                elasticity_sku_effect = pm.Deterministic(
                    'elasticity_sku_effect',
                    elasticity_sku_raw * self.sku_effect_std
                )
                
                # Compute elasticity for each SKU - use product_class_idx directly
                # product_class_idx already maps each SKU to its product class
                elasticity_sku = pm.Deterministic(
                    'elasticity_sku',
                    elasticity_mu + elasticity_class_effect[product_class_idx] + 
                    elasticity_sku_effect
                )
                
                # Calculate elasticity per observation
                elasticity = pm.Deterministic(
                    'elasticity',
                    elasticity_mu + 
                    elasticity_class_effect[product_class_idx[sku_idx]] + 
                    elasticity_sku_effect[sku_idx]
                )
                
                # Intercepts
                intercept_mu = pm.Normal('intercept_mu', mu=5.0, sigma=1.0)
                intercept_class = pm.Normal('intercept_class', mu=0, sigma=0.5, shape=n_product_classes)
                intercept_sku_raw = pm.Normal('intercept_sku_raw', mu=0, sigma=1.0, shape=n_skus)
                intercept_sku_effect = pm.Deterministic(
                    'intercept_sku_effect',
                    intercept_sku_raw * 0.5
                )
                
                # Compute intercept for each observation
                intercept = pm.Deterministic(
                    'intercept',
                    intercept_mu + 
                    intercept_class[product_class_idx[sku_idx]] + 
                    intercept_sku_effect[sku_idx]
                )
                
                # Seasonality effects if seasonal features are included
                if seasonal_X is not None:
                    season_coef = pm.Normal(
                        'season_coef', 
                        mu=0, 
                        sigma=0.1, 
                        shape=seasonal_X.shape[1]
                    )
                    seasonal_effect = pm.math.dot(seasonal_X, season_coef)
                else:
                    seasonal_effect = 0
                
                # Model error (observation noise)
                sigma = pm.HalfNormal('sigma', sigma=0.5)
                
                # Expected value
                mu = intercept + elasticity * X + seasonal_effect
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            logger.info(f"Successfully built model")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelBuildError(f"Error building model: {str(e)}")
    
    def get_model(self) -> Any:
        """
        Get the built model.
        
        Returns:
            PyMC model or None if not built
        """
        return self._model 

    def _prepare_model_data(self, data_input: Union[pd.DataFrame, ModelData]) -> ModelData:
        """
        Convert DataFrame to ModelData object for the model.
        
        Args:
            data_input: Input data as DataFrame or ModelData
            
        Returns:
            ModelData object ready for modeling
            
        Raises:
            ModelBuildError: If data preparation fails
        """
        # If already a ModelData object, return as is
        if isinstance(data_input, ModelData):
            return data_input
            
        # Convert DataFrame to ModelData
        try:
            # Extract data required for the model
            df = data_input
            
            # Get required columns
            price_col = "price"  # We standardized these names in BayesianModel
            quantity_col = "quantity"
            sku_col = "sku"  
            product_class_col = "product_class"
            
            # Check for log-transformed columns
            log_price_col = f"log_{price_col}"
            log_quantity_col = f"log_{quantity_col}"
            
            # Use log-transformed columns if available
            X_col = log_price_col if log_price_col in df.columns else price_col
            y_col = log_quantity_col if log_quantity_col in df.columns else quantity_col
            
            # Extract features and target
            X = df[X_col].values
            y = df[y_col].values
            
            # Create SKU and product class indices
            skus = df[sku_col].unique()
            n_skus = len(skus)
            sku_to_idx = {sku: i for i, sku in enumerate(skus)}
            sku_idx = np.array([sku_to_idx[sku] for sku in df[sku_col]])
            
            product_classes = df[product_class_col].unique()
            n_classes = len(product_classes)
            class_to_idx = {cls: i for i, cls in enumerate(product_classes)}
            
            # Create product class index for each SKU
            sku_class_map = df[[sku_col, product_class_col]].drop_duplicates()
            sku_to_class = dict(zip(sku_class_map[sku_col], sku_class_map[product_class_col]))
            product_class_idx = np.array([class_to_idx[sku_to_class[sku]] for sku in skus])
            
            # Extract seasonality features if available
            seasonality_cols = [col for col in df.columns if col.startswith(('month_', 'holiday_', 'event_'))]
            seasonality_features = df[seasonality_cols].values if seasonality_cols else None
            
            # Create ModelData object
            logger.info(f"Created ModelData from DataFrame with {n_skus} SKUs and {n_classes} product classes")
            return ModelData(
                X=X,
                y=y,
                sku_idx=sku_idx,
                product_class_idx=product_class_idx,
                n_skus=n_skus,
                n_classes=n_classes,
                seasonality_features=seasonality_features,
                sku_names=skus,
                class_names=product_classes
            )
            
        except Exception as e:
            raise ModelBuildError(f"Error preparing model data: {str(e)}") 

    def _check_required_columns(self, data: pd.DataFrame, required_cols: List[str]) -> None:
        """
        Check if the DataFrame contains all the required columns.
        
        Args:
            data: DataFrame to check
            required_cols: List of required column names
            
        Raises:
            ModelBuildError: If any required column is missing
        """
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ModelBuildError(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values in required columns
        for col in required_cols:
            if data[col].isnull().any():
                raise ModelBuildError(f"Column '{col}' contains NaN values")