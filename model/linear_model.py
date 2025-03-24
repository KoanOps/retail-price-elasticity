"""
Linear Regression Model Implementation for Price Elasticity Estimation.

This module provides a simple linear regression implementation for elasticity
estimation, following the same interface as the BayesianModel for compatibility.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from utils.logging_utils import get_logger, log_step
from model.exceptions import ModelError, DataError, FittingError
from model.base_model import BaseElasticityModel

# Try to import scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    sklearn_available = True
except ImportError:
    sklearn_available = False
    print("scikit-learn not available. LinearRegressionModel will not work.")

# Setup logging
logger = get_logger()

class LinearRegressionModel(BaseElasticityModel):
    """
    Simple linear regression model for elasticity estimation.
    
    This class implements a log-log linear regression model for estimating
    price elasticity. It follows the same interface as the BayesianModel.
    """
    
    def __init__(
        self,
        results_dir: Union[str, Path],
        model_name: str = "linear_elasticity_model",
        data_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        use_seasonality: bool = True,
        **kwargs
    ):
        """
        Initialize the linear regression model for elasticity estimation.
        
        Args:
            results_dir: Directory to save results
            model_name: Name of the model
            data_config: Configuration for data preparation
            model_config: Configuration for model building
            use_seasonality: Whether to use seasonal effects (ignored in linear model)
            **kwargs: Additional parameters (ignored)
            
        Raises:
            ModelError: If initialization fails
        """
        # Call parent class constructor with required parameters
        super().__init__(
            results_dir=str(results_dir), 
            model_name=model_name,
            data_config=data_config,
            model_config=model_config
        )
        
        if not sklearn_available:
            raise ModelError("scikit-learn not available. Cannot create linear regression model.")
            
        # Store configuration
        self.use_seasonality = use_seasonality
        
        # Initialize model state
        self.linear_models = {}  # One model per SKU
        self.elasticity_estimates = None
        self.is_built = False
            
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for modeling.
        
        Args:
            data: Input DataFrame with raw data
            
        Returns:
            Prepared DataFrame with log-transformed features
            
        Raises:
            DataError: If preparation fails
        """
        try:
            # Copy the input data
            prepared_data = data.copy()
            
            # Ensure we have the required columns
            required_cols = ['sku', 'price', 'quantity']
            missing_cols = [col for col in required_cols if col not in prepared_data.columns]
            
            if missing_cols:
                # Try case-insensitive column mapping
                for col in missing_cols:
                    col_candidates = [c for c in prepared_data.columns if c.lower() == col.lower()]
                    if col_candidates:
                        prepared_data[col] = prepared_data[col_candidates[0]]
                
                # Check again for missing columns
                missing_cols = [col for col in required_cols if col not in prepared_data.columns]
                if missing_cols:
                    raise DataError(f"Missing required columns: {missing_cols}")
            
            # Log-transform price and quantity if needed
            if 'log_price' not in prepared_data.columns:
                prepared_data['log_price'] = np.log(prepared_data['price'])
                logger.info("Created log-transformed price column: log_price")
                
            if 'log_quantity' not in prepared_data.columns:
                prepared_data['log_quantity'] = np.log(prepared_data['quantity'])
                logger.info("Created log-transformed quantity column: log_quantity")
            
            logger.info(f"Data preparation complete. Shape: {prepared_data.shape}")
            self.prepared_data = prepared_data
            return prepared_data
            
        except Exception as e:
            error_msg = f"Error during data preparation: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg)
            
    def build_model(self) -> bool:
        """
        Build the linear model (creates LinearRegression instances).
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            FittingError: If model building fails
        """
        try:
            if not hasattr(self, 'prepared_data') or self.prepared_data is None:
                raise FittingError("No prepared data available. Call prepare_data first.")
                
            logger.info("Building linear regression models (one per SKU)...")
            
            # Nothing to actually build - we'll create models per SKU during fitting
            self.is_built = True
            
            logger.info("Linear regression model ready")
            return True
            
        except Exception as e:
            logger.error(f"Model building failed: {str(e)}")
            self.is_built = False
            raise FittingError(f"Model building failed: {str(e)}")
    
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Fit the linear regression models to the data.
        
        Args:
            data: Input DataFrame with raw data
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Dictionary with model results
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            # Prepare data
            prepared_data = self.prepare_data(data)
            
            # Build model
            self.build_model()
            
            # Check if promotional data is available
            has_promo_feature = 'Is_Promo' in prepared_data.columns
            if has_promo_feature:
                logger.info("Promotional data detected - incorporating into linear model")
            
            # Fit a separate linear regression model for each SKU
            unique_skus = prepared_data['sku'].unique()
            logger.info(f"Fitting {len(unique_skus)} SKU-level linear regression models")
            
            elasticities = {}
            self.linear_models = {}
            
            for sku in unique_skus:
                # Get data for this SKU
                sku_data = prepared_data[prepared_data['sku'] == sku]
                
                if len(sku_data) < 5:  # Skip SKUs with too few observations
                    logger.warning(f"Skipping SKU {sku} with only {len(sku_data)} observations")
                    continue
                
                # Create feature matrix and target vector
                if has_promo_feature:
                    # Include price, promo flag, and interaction term
                    X = pd.DataFrame()
                    X['log_price'] = sku_data['log_price']
                    X['Is_Promo'] = sku_data['Is_Promo']
                    # Add interaction term (log_price * Is_Promo)
                    X['log_price_promo'] = X['log_price'] * X['Is_Promo']
                else:
                    # Just use price as the predictor
                    X = sku_data[['log_price']]
                    
                y = sku_data['log_quantity']
                
                # Create and fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Store the model
                self.linear_models[sku] = {
                    'model': model,
                    'has_promo': has_promo_feature
                }
                
                # Store elasticity estimate
                if has_promo_feature:
                    # Base elasticity is the coefficient of log_price
                    base_elasticity = float(model.coef_[0])
                    # Promotional elasticity is base + interaction term
                    promo_elasticity = float(model.coef_[2])
                    # Store both for reference
                    elasticities[sku] = {
                        'base': base_elasticity,
                        'promo': promo_elasticity,
                        # Weighted average based on proportion of promo observations
                        'weighted': float(np.average([base_elasticity, promo_elasticity], 
                                                    weights=[1-sku_data['Is_Promo'].mean(), 
                                                             sku_data['Is_Promo'].mean()]))
                    }
                else:
                    # Simple case: elasticity is just the coefficient of log_price
                    elasticities[sku] = float(model.coef_[0])
            
            # Store elasticity estimates
            self.elasticity_estimates = elasticities
            
            # Compute summary statistics
            if has_promo_feature:
                # Get the weighted average elasticities for summary
                mean_elasticity = np.mean([e['weighted'] for e in elasticities.values()])
                median_elasticity = np.median([e['weighted'] for e in elasticities.values()])
                
                # Also compute average base and promo elasticities
                mean_base = np.mean([e['base'] for e in elasticities.values()])
                mean_promo = np.mean([e['promo'] for e in elasticities.values()])
                
                logger.info(f"Mean base elasticity: {mean_base:.4f}")
                logger.info(f"Mean promo elasticity: {mean_promo:.4f}")
            else:
                mean_elasticity = np.mean(list(elasticities.values()))
                median_elasticity = np.median(list(elasticities.values()))
            
            logger.info(f"Fitted {len(elasticities)} SKU-level models")
            logger.info(f"Mean elasticity: {mean_elasticity:.4f}")
            logger.info(f"Median elasticity: {median_elasticity:.4f}")
            
            # Return results
            return {
                "elasticities": elasticities,
                "summary": {
                    "mean_elasticity": mean_elasticity,
                    "median_elasticity": median_elasticity,
                    "num_skus": len(elasticities),
                    "has_promo_feature": has_promo_feature
                }
            }
            
        except Exception as e:
            error_msg = f"Error during model fitting: {str(e)}"
            logger.error(error_msg)
            raise FittingError(error_msg)
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained models.
        
        Args:
            X: DataFrame with input features
            
        Returns:
            DataFrame with predictions
            
        Raises:
            ModelError: If prediction fails
        """
        if not self.linear_models:
            raise ModelError("No models available. Call fit() first.")
            
        try:
            # Create a copy of the input data
            predictions = X.copy()
            
            # Add log_price column if not present
            if 'log_price' not in predictions.columns and 'price' in predictions.columns:
                predictions['log_price'] = np.log(predictions['price'])
            
            # Make predictions for each SKU
            predicted_quantities = []
            
            for _, row in predictions.iterrows():
                sku = row['sku']
                log_price = row['log_price']
                
                if sku in self.linear_models:
                    # Use the appropriate SKU model
                    model = self.linear_models[sku]['model']
                    if self.linear_models[sku]['has_promo']:
                        # Use the promo model
                        log_qty_pred = model.predict([[log_price, 1, log_price * 1]])[0]
                    else:
                        # Use the base model
                        log_qty_pred = model.predict([[log_price]])[0]
                    qty_pred = np.exp(log_qty_pred)
                else:
                    # No model for this SKU
                    qty_pred = np.nan
                    
                predicted_quantities.append(qty_pred)
                
            predictions['predicted_quantity'] = predicted_quantities
            
            return predictions
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg)

    def estimate_elasticities(self) -> Dict[str, float]:
        """
        Estimate price elasticities for each SKU.
        
        Returns:
            Dictionary mapping SKU IDs to elasticity estimates
            
        Raises:
            ModelError: If elasticities cannot be estimated
        """
        if not self.linear_models:
            raise ModelError("No trained models available. Call fit() first.")
        
        if self.elasticity_estimates is not None:
            return self.elasticity_estimates
        
        # For linear models, elasticities are simply the coefficients
        # of log_price in the regression
        elasticities = {}
        
        for sku, model_info in self.linear_models.items():
            model = model_info['model']
            if model_info['has_promo']:
                # Use the promo model
                log_price = self.prepared_data['log_price']
                log_qty = self.prepared_data['log_quantity']
                promo_elasticity = float(model.coef_[2])
                base_elasticity = float(model.coef_[0])
                # Promotional elasticity is base + interaction term
                promo_elasticity = float(model.coef_[2])
                # Store both for reference
                elasticities[sku] = {
                    'base': base_elasticity,
                    'promo': promo_elasticity,
                    # Weighted average based on proportion of promo observations
                    'weighted': float(np.average([base_elasticity, promo_elasticity], 
                                                weights=[1-self.prepared_data['Is_Promo'].mean(), 
                                                         self.prepared_data['Is_Promo'].mean()]))
                }
            else:
                # Simple case: elasticity is just the coefficient of log_price
                elasticities[sku] = float(model.coef_[0])
            
        self.elasticity_estimates = elasticities
        
        # Print summary
        if self.linear_models:
            if self.linear_models[list(self.linear_models.keys())[0]]['has_promo']:
                # Get the weighted average elasticities for summary
                mean_elasticity = np.mean([e['weighted'] for e in elasticities.values()])
                median_elasticity = np.median([e['weighted'] for e in elasticities.values()])
                
                # Also compute average base and promo elasticities
                mean_base = np.mean([e['base'] for e in elasticities.values()])
                mean_promo = np.mean([e['promo'] for e in elasticities.values()])
                
                logger.info(f"Mean base elasticity: {mean_base:.4f}")
                logger.info(f"Mean promo elasticity: {mean_promo:.4f}")
            else:
                mean_elasticity = np.mean(list(elasticities.values()))
                median_elasticity = np.median(list(elasticities.values()))
            
            logger.info(f"Estimated elasticities for {len(elasticities)} SKUs")
            logger.info(f"Mean elasticity: {mean_elasticity:.4f}")
            logger.info(f"Median elasticity: {median_elasticity:.4f}")
        
        return elasticities 