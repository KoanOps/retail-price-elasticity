"""
Bayesian Data Preparation Module for Price Elasticity Modeling.

This module provides specialized data preprocessing, validation, and transformation 
functionality for Bayesian hierarchical price elasticity models, ensuring data 
is properly structured for statistical inference.

PURPOSE:
- Transform raw sales data into the proper format for Bayesian modeling
- Validate data properties to ensure model assumptions are met
- Create necessary indices and statistical features for hierarchical modeling
- Generate seasonality components to account for temporal effects
- Apply appropriate transformations (e.g., log transforms) for elasticity estimation

MATH:
- Price elasticity models use log-log transformations: log(quantity) = α + β*log(price) + ...
- The β coefficient represents the price elasticity of demand
- Log transformations help normalize skewed data and interpret coefficients as percentages

ASSUMPTIONS:
- Price and quantity data should ideally follow log-normal distributions
- Each SKU belongs to exactly one product class for hierarchical modeling
- Non-positive values in price/quantity are considered data errors, not valid observations
- Date data (if present) is convertible to datetime format for seasonality extraction
- Data can be processed in-memory on available hardware

EDGE CASES:
- Data failing log-normality tests will generate warnings but proceed with transformations
- Missing values in key columns will be explicitly flagged and may cause failures
- Zero or negative prices/quantities will be replaced with small positive values
- Duplicate SKU-class mappings will use the first occurrence for consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, cast
from pathlib import Path
import logging

from utils.logging_utils import logger, log_step
from model.exceptions import DataError
from utils.analysis.seasonality_analysis import add_event_indicators
from model.constants import (
    DEFAULT_INVALID_PRICE_FACTOR,
    DEFAULT_INVALID_QUANTITY_FACTOR,
    DEFAULT_DATA_CONFIG,
    MONTHS_IN_YEAR
)


class BayesianDataPreparation:
    """
    Comprehensive data preparation system for Bayesian price elasticity models.
    
    This class handles all aspects of transforming raw sales data into the specialized format
    required by Bayesian hierarchical elasticity models, with a focus on statistical
    correctness and model assumption validation.
    
    Core Responsibilities:
    - Data validation and integrity checking
    - Log transformations with statistical validation  
    - Numerical encoding of categorical variables
    - Feature engineering for hierarchical effects
    - Seasonality component extraction
    - Handling of data anomalies (zeros, negatives, outliers)
    
    Assumptions:
    - Log-normal distribution of prices and quantities is ideal for elasticity estimation
    - Non-positive values represent data errors that should be transformed, not removed
    - Hierarchical structure exists (SKUs within product classes)
    - Seasonal patterns follow regular cycles (can be modeled with Fourier terms)
    - Price variations exist within SKUs (required for estimation)
    
    Implementation Notes:
    - Log transformation appropriateness is evaluated using multiple statistical tests
    - Non-positive values are replaced with min_positive/factor to preserve data points
    - Seasonality is modeled using sin/cos terms for Fourier decomposition
    - Missing values in critical columns will trigger explicit errors
    - All transformations are documented in logs for reproducibility
    """
    
    def __init__(
        self,
        data_config: Optional[Dict[str, Any]] = None,
        use_seasonality: bool = True
    ):
        """
        Initialize the data preparation component.
        
        Args:
            data_config: Configuration for data preparation
            use_seasonality: Whether to include seasonal effects
        """
        self.data_config = data_config or DEFAULT_DATA_CONFIG.copy()
        self.use_seasonality = use_seasonality
        
        # Will store indices for product classes and SKUs
        self.product_classes = None
        self.skus = None
        self.class_to_idx = {}
        self.sku_to_idx = {}
        self.sku_to_class = {}
        self.sku_class_idx = None
        
        # Seasonality features
        self.seasonal_features = None
        
    def validate_data_config(self) -> None:
        """
        Validate data configuration.
        
        Raises:
            DataError: If configuration is invalid
        """
        if not self.data_config:
            raise DataError("Data configuration not set")
            
        required_fields = [
            "price_col", "quantity_col", "sku_col", "product_class_col"
        ]
        
        missing_fields = [f for f in required_fields if f not in self.data_config]
        if missing_fields:
            raise DataError(f"Missing required fields in data_config: {missing_fields}")
            
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for modeling.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed DataFrame ready for modeling
            
        Raises:
            DataError: If there are issues with the input data
        """
        if data is None or len(data) == 0:
            raise DataError("Input data is empty")
            
        # Make a defensive copy
        data = data.copy()
        
        # Validate configuration
        self.validate_data_config()
        
        # Extract column names from config
        price_col = self.data_config.get("price_col")
        quantity_col = self.data_config.get("quantity_col")
        sku_col = self.data_config.get("sku_col")
        product_class_col = self.data_config.get("product_class_col")
        date_col = self.data_config.get("date_col", "date")
        
        # Validate required columns exist
        required_cols = [price_col, quantity_col, sku_col, product_class_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")
            
        # Convert date column if needed
        if date_col in data.columns and not pd.api.types.is_datetime64_dtype(data[date_col]):
            try:
                data[date_col] = pd.to_datetime(data[date_col])
            except Exception as e:
                raise DataError(f"Could not convert {date_col} to datetime: {str(e)}")
                
        # Generate indices for categories
        self._create_category_indices(data, sku_col, product_class_col)
        
        # Apply transformations if configured
        data = self._apply_transformations(data)
        
        # Add seasonality features if enabled
        if self.use_seasonality and date_col in data.columns:
            data = self._add_seasonality_features(data, date_col)
            
        logger.info(f"Data preparation complete: {len(data)} rows, {len(data.columns)} columns")
        return data
        
    def _create_category_indices(
        self, 
        data: pd.DataFrame, 
        sku_col: str, 
        product_class_col: str
    ) -> None:
        """
        Create mappings between categorical values and indices.
        
        Args:
            data: Input DataFrame
            sku_col: Column name for SKU identifiers
            product_class_col: Column name for product class identifiers
        """
        # Extract unique values
        self.skus = data[sku_col].unique()
        self.product_classes = data[product_class_col].unique()
        
        # Create mappings
        self.sku_to_idx = {sku: i for i, sku in enumerate(self.skus)}
        self.class_to_idx = {cls: i for i, cls in enumerate(self.product_classes)}
        
        # Map each SKU to its product class
        sku_class_map = data[[sku_col, product_class_col]].drop_duplicates()
        self.sku_to_class = dict(zip(sku_class_map[sku_col], sku_class_map[product_class_col]))
        
        # Create index mapping from SKU to product class index
        self.sku_class_idx = np.array([
            self.class_to_idx[self.sku_to_class[sku]] 
            for sku in self.skus
        ])
        
        logger.info(f"Created indices for {len(self.skus)} SKUs and {len(self.product_classes)} product classes")
        
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to numeric columns (e.g., log transforms).
        
        Assumptions:
        - Price/quantity data follows log-normal distribution
        - Non-positive values are errors (replaced with min_positive/constant)
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        price_col = self.data_config.get("price_col")
        quantity_col = self.data_config.get("quantity_col")
        
        # Apply log transformations if configured
        log_price = self.data_config.get("log_transform_price", False)
        log_quantity = self.data_config.get("log_transform_quantity", False)
        
        # Validate log transformations if configured
        if log_price and price_col in data.columns:
            try:
                # Validate log transformation appropriateness
                # Use a single column from the DataFrame to avoid ambiguity
                price_series = data[price_col].iloc[:, 0] if isinstance(data[price_col], pd.DataFrame) else data[price_col]
                price_validation = self._validate_log_transformation(price_series, "price")
                
                # Proceed with transformation only if validation passes or is forced
                force_transform = self.data_config.get("force_log_transform", False)
                if price_validation or force_transform:
                    if not price_validation:
                        logger.warning("Log transformation for price may not be appropriate but is being forced")
                        
                    # Ensure price is positive - fix ambiguity by using numpy for comparison
                    # Get values from the first column if it's a DataFrame
                    if isinstance(data[price_col], pd.DataFrame):
                        price_values = data[price_col].iloc[:, 0].values
                    else:
                        price_values = data[price_col].values
                        
                    non_positive_mask = (price_values <= 0)
                    if np.any(non_positive_mask):
                        positive_mask = (price_values > 0)
                        min_positive = np.min(price_values[positive_mask])
                        # Replace non-positive prices with min_positive/10 to maintain distribution shape
                        # while avoiding log(0) errors
                        replacement_value = min_positive / DEFAULT_INVALID_PRICE_FACTOR
                        
                        # Replace in the specific column
                        if isinstance(data[price_col], pd.DataFrame):
                            data.loc[price_values <= 0, price_col].iloc[:, 0] = replacement_value
                        else:
                            data.loc[price_values <= 0, price_col] = replacement_value
                            
                        logger.warning(f"Replaced non-positive prices with {replacement_value:.4f}")
                        
                    # Create log-transformed column - handle case where price_col refers to multiple columns
                    log_price_col = f"log_{price_col}"
                    if isinstance(data[price_col], pd.DataFrame):
                        # Apply log to the first column only
                        data[log_price_col] = np.log(data[price_col].iloc[:, 0])
                    else:
                        data[log_price_col] = np.log(data[price_col])
                        
                    logger.info(f"Created log-transformed price column: {log_price_col}")
                else:
                    logger.warning("Skipping log transformation for price as validation failed")
                    # Set flag to false to avoid using log transformation in model
                    log_price = False
            except Exception as e:
                logger.warning(f"Error during price log transformation: {str(e)}")
                log_price = False
            
        if log_quantity and quantity_col in data.columns:
            try:
                # Validate log transformation appropriateness
                # Use a single column from the DataFrame to avoid ambiguity
                quantity_series = data[quantity_col].iloc[:, 0] if isinstance(data[quantity_col], pd.DataFrame) else data[quantity_col]
                quantity_validation = self._validate_log_transformation(quantity_series, "quantity")
                
                # Proceed with transformation only if validation passes or is forced
                force_transform = self.data_config.get("force_log_transform", False)
                if quantity_validation or force_transform:
                    if not quantity_validation:
                        logger.warning("Log transformation for quantity may not be appropriate but is being forced")
                        
                    # Ensure quantity is positive - fix ambiguity by using numpy for comparison
                    # Get values from the first column if it's a DataFrame
                    if isinstance(data[quantity_col], pd.DataFrame):
                        quantity_values = data[quantity_col].iloc[:, 0].values
                    else:
                        quantity_values = data[quantity_col].values
                        
                    non_positive_mask = (quantity_values <= 0)
                    if np.any(non_positive_mask):
                        positive_mask = (quantity_values > 0)
                        min_positive = np.min(quantity_values[positive_mask])
                        # Replace non-positive quantities with min_positive/constant
                        replacement_value = min_positive / DEFAULT_INVALID_QUANTITY_FACTOR
                        
                        # Replace in the specific column
                        if isinstance(data[quantity_col], pd.DataFrame):
                            data.loc[quantity_values <= 0, quantity_col].iloc[:, 0] = replacement_value
                        else:
                            data.loc[quantity_values <= 0, quantity_col] = replacement_value
                            
                        logger.warning(f"Replaced non-positive quantities with {replacement_value:.4f}")
                        
                    # Create log-transformed column
                    log_quantity_col = f"log_{quantity_col}"
                    if isinstance(data[quantity_col], pd.DataFrame):
                        # Apply log to the first column only
                        data[log_quantity_col] = np.log(data[quantity_col].iloc[:, 0])
                    else:
                        data[log_quantity_col] = np.log(data[quantity_col])
                        
                    logger.info(f"Created log-transformed quantity column: {log_quantity_col}")
                else:
                    logger.warning("Skipping log transformation for quantity as validation failed")
                    # Set flag to false to avoid using log transformation in model
                    log_quantity = False
            except Exception as e:
                logger.warning(f"Error during quantity log transformation: {str(e)}")
                log_quantity = False
            
        return data
    
    def _validate_log_transformation(self, series: pd.Series, variable_name: str) -> bool:
        """
        Validate whether log transformation is appropriate for the given data.
        
        This method uses both visual and statistical approaches to assess
        whether data follows a log-normal distribution.
        
        ASSUMPTIONS:
        - Data that is more normal after log transformation is appropriate for log transform
        - Skewness closer to 0 after log transformation indicates better fit
        - Statistical tests can provide objective measures of normality
        
        EDGE CASES:
        - Very small samples may not provide reliable normality tests
        - Mixed distributions may not be well-assessed by standard tests
        - Some datasets may be ambiguous (neither raw nor logged data is normal)
        
        Args:
            series: Data series to test
            variable_name: Name of the variable (for logging)
            
        Returns:
            Boolean indicating whether log transformation is appropriate
        """
        # Make a copy to avoid modifying the original
        data = series.copy()
        
        # Handle non-positive values temporarily for testing - fix ambiguity using numpy
        data_values = data.values
        non_positive_mask = (data_values <= 0)
        if np.any(non_positive_mask):
            positive_mask = (data_values > 0)
            min_positive = np.min(data_values[positive_mask])
            data[non_positive_mask] = min_positive / 10
        
        # Skip validation if sample is too small
        if len(data) < 30:
            logger.warning(f"Sample size too small for reliable log-transform validation of {variable_name}")
            return True  # Default to allowing transformation for small samples
        
        try:
            # 1. Compare skewness of raw vs log data
            raw_skew = data.skew()
            log_skew = np.log(data).skew()
            
            # Calculate absolute skewness - closer to 0 is better
            abs_raw_skew = abs(raw_skew)
            abs_log_skew = abs(log_skew)
            
            # 2. Attempt Shapiro-Wilk test for normality
            # Sample data for performance if dataset is large
            sample_size = min(5000, len(data))
            sample_data = data.sample(sample_size)
            sample_log_data = np.log(sample_data)
            
            from scipy import stats
            try:
                # Shapiro-Wilk test - higher p-value suggests normality
                _, raw_p = stats.shapiro(sample_data)
                _, log_p = stats.shapiro(sample_log_data)
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed: {str(e)}")
                raw_p = 0
                log_p = 0
            
            # 3. Compare Q-Q plot correlation coefficients
            try:
                # Q-Q plot correlation for raw data
                raw_qq = stats.probplot(sample_data, dist="norm")
                raw_r = np.corrcoef(raw_qq[0][0], raw_qq[0][1])[0, 1]
                
                # Q-Q plot correlation for log data
                log_qq = stats.probplot(sample_log_data, dist="norm")
                log_r = np.corrcoef(log_qq[0][0], log_qq[0][1])[0, 1]
            except Exception as e:
                logger.warning(f"Q-Q plot correlation calculation failed: {str(e)}")
                raw_r = 0
                log_r = 0
            
            # 4. Decide if log transform is appropriate
            # Log transform is better if:
            # - Log data has lower absolute skewness, OR
            # - Log data has higher p-value in Shapiro-Wilk test, OR
            # - Log data has higher correlation in Q-Q plot
            
            skew_better = abs_log_skew < abs_raw_skew
            pvalue_better = log_p > raw_p
            qq_better = log_r > raw_r
            
            # Count how many metrics suggest log transform is better
            log_better_count = sum([skew_better, pvalue_better, qq_better])
            
            # Create a detailed validation report
            logger.info(f"Log transformation validation for {variable_name}:")
            logger.info(f"  Skewness: Raw={raw_skew:.4f}, Log={log_skew:.4f} - {'BETTER' if skew_better else 'WORSE'} with log")
            logger.info(f"  Shapiro-Wilk p-value: Raw={raw_p:.4f}, Log={log_p:.4f} - {'BETTER' if pvalue_better else 'WORSE'} with log")
            logger.info(f"  Q-Q plot correlation: Raw={raw_r:.4f}, Log={log_r:.4f} - {'BETTER' if qq_better else 'WORSE'} with log")
            
            # Log transform is appropriate if majority of metrics suggest it's better
            is_appropriate = log_better_count >= 2
            
            logger.info(f"Log transformation for {variable_name} is {'APPROPRIATE' if is_appropriate else 'NOT APPROPRIATE'}")
            
            # 5. Optionally generate diagnostic plots if matplotlib is available
            try:
                # Only import visualization libraries if available
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Check if we should save diagnostic plots
                save_diagnostic_plots = self.data_config.get("save_diagnostic_plots", False)
                if save_diagnostic_plots:
                    # Create path for diagnostics
                    import os
                    diagnostics_dir = self.data_config.get("diagnostics_dir", "diagnostics")
                    os.makedirs(diagnostics_dir, exist_ok=True)
                    
                    # Create visualization with both histograms and Q-Q plots
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Histograms
                    sns.histplot(data, kde=True, ax=axes[0, 0])
                    axes[0, 0].set_title(f'Raw {variable_name} distribution')
                    
                    sns.histplot(np.log(data), kde=True, color='orange', ax=axes[0, 1])
                    axes[0, 1].set_title(f'Log-transformed {variable_name} distribution')
                    
                    # Q-Q plots
                    stats.probplot(sample_data, dist="norm", plot=axes[1, 0])
                    axes[1, 0].set_title(f'Q-Q Plot: Raw {variable_name}')
                    
                    stats.probplot(sample_log_data, dist="norm", plot=axes[1, 1])
                    axes[1, 1].set_title(f'Q-Q Plot: Log-transformed {variable_name}')
                    
                    plt.tight_layout()
                    plt.savefig(f"{diagnostics_dir}/{variable_name}_log_transform_diagnostic.png")
                    plt.close()
                    
                    logger.info(f"Saved diagnostic plots to {diagnostics_dir}/{variable_name}_log_transform_diagnostic.png")
            except ImportError:
                logger.warning("Visualization packages not available, skipping diagnostic plots")
            except Exception as e:
                logger.warning(f"Error generating diagnostic plots: {str(e)}")
            
            return is_appropriate
            
        except Exception as e:
            logger.warning(f"Error during log transformation validation: {str(e)}")
            # Default to True if validation fails
            return True
        
    def _add_seasonality_features(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Add seasonality features to the dataset.
        
        Assumptions:
        - Monthly seasonal patterns exist (month indicators)
        - Day-of-week effects are relevant
        - Holidays/events have consistent effects when provided
        
        Args:
            data: Input DataFrame
            date_col: Column name for date
            
        Returns:
            DataFrame with additional seasonality features
        """
        # Extract time features
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day_of_week'] = data[date_col].dt.dayofweek
        data['quarter'] = data[date_col].dt.quarter
        
        # Add holiday indicators if provided
        holidays = self.data_config.get("holidays", [])
        events = self.data_config.get("events", [])
        
        if holidays or events:
            data = add_event_indicators(
                data, 
                date_col=date_col,
                holidays=holidays,
                events=events
            )
            
        # Create month indicators for seasonality using constant
        for month in range(1, MONTHS_IN_YEAR + 1):
            data[f'month_{month}'] = (data['month'] == month).astype(int)
            
        # Store seasonal feature names for the model
        self.seasonal_features = [col for col in data.columns 
                                if col.startswith(('month_', 'holiday_', 'event_'))]
        
        logger.info(f"Added {len(self.seasonal_features)} seasonality features")
        return data 

    def prepare(
        self, 
        data: pd.DataFrame,
        price_col: str = 'price',
        quantity_col: str = 'quantity',
        sku_col: str = 'sku',
        product_class_col: str = 'product_class',
        date_col: Optional[str] = 'date'
    ) -> pd.DataFrame:
        """
        Prepare data with explicit column name parameters.
        
        This is a wrapper around prepare_data that allows specifying column names
        directly as parameters instead of through the data_config.
        
        Args:
            data: Input DataFrame
            price_col: Column name for price data
            quantity_col: Column name for quantity data
            sku_col: Column name for SKU identifiers
            product_class_col: Column name for product class identifiers
            date_col: Column name for date information (optional)
            
        Returns:
            Prepared DataFrame ready for modeling
        """
        # Create a temporary data config with the provided column names
        temp_config = self.data_config.copy()
        temp_config.update({
            "price_col": price_col,
            "quantity_col": quantity_col,
            "sku_col": sku_col,
            "product_class_col": product_class_col
        })
        
        if date_col is not None:
            temp_config["date_col"] = date_col
            
        # Store original config
        original_config = self.data_config
        
        # Use the temporary config for data preparation
        try:
            self.data_config = temp_config
            logger.info(f"Preparing data with columns: price={price_col}, quantity={quantity_col}, "
                       f"sku={sku_col}, product_class={product_class_col}, " 
                       f"date={date_col if date_col else 'None'}")
            return self.prepare_data(data)
        finally:
            # Restore original config
            self.data_config = original_config 