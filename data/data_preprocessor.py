#!/usr/bin/env python3
"""
Data Preprocessor for Retail Price Elasticity Analysis

This module provides data preprocessing functionality for retail data analysis,
focusing on cleaning, transforming, and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

# Use absolute imports
from utils.logging_utils import logger


class DataPreprocessor:
    """
    Handles data preprocessing for retail price elasticity analysis.
    
    This class is responsible for:
    - Data cleaning and validation
    - Feature engineering
    - Date feature extraction
    - SKU-specific feature creation
    
    Parameters
    ----------
    datetime_col : str, optional
        Name of the column containing datetime information.
    price_col : str
        Name of the column containing price information.
    quantity_col : str
        Name of the column containing quantity information.
    sku_col : str
        Name of the column containing SKU identifiers.
    product_class_col : str, optional
        Name of the column containing product class information.
    """
    
    def __init__(
        self,
        datetime_col: Optional[str] = "Date",
        price_col: str = "Price_Per_Unit",
        quantity_col: str = "Qty_Sold",
        sku_col: str = "SKU_Coded",
        product_class_col: Optional[str] = "Product_Class_Code"
    ):
        self.datetime_col = datetime_col
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.sku_col = sku_col
        self.product_class_col = product_class_col
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard preprocessing steps to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data to preprocess.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed data.
        """
        logger.info(f"Preprocessing data, initial shape: {data.shape}")
        
        # Calculate Price_Per_Unit if not present
        if self.price_col not in data.columns and 'Total_Sale_Value' in data.columns and self.quantity_col in data.columns:
            data[self.price_col] = data['Total_Sale_Value'] / data[self.quantity_col]
            logger.info(f"Calculated {self.price_col} from Total_Sale_Value and {self.quantity_col}")
        
        # Drop rows with missing values
        data = data.dropna()
        
        # Remove rows with zero or negative prices
        if self.price_col in data.columns:
            data = data[data[self.price_col] > 0]
        
        # Remove rows with zero or negative quantities
        if self.quantity_col in data.columns:
            data = data[data[self.quantity_col] > 0]
        
        logger.info(f"Preprocessing complete, final shape: {data.shape}")
        return data
    
    def add_date_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date features from the datetime column.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing the datetime column.
        
        Returns
        -------
        pd.DataFrame
            Data with added date features.
        """
        if not self.datetime_col or self.datetime_col not in data.columns:
            logger.warning(f"Datetime column not found, skipping date features")
            return data
            
        logger.info("Adding date features")
        
        if not pd.api.types.is_datetime64_any_dtype(data[self.datetime_col]):
            logger.warning(f"Column {self.datetime_col} is not datetime type, attempting conversion")
            try:
                data[self.datetime_col] = pd.to_datetime(data[self.datetime_col])
            except:
                logger.error(f"Failed to convert {self.datetime_col} to datetime, skipping date features")
                return data
        
        # Extract basic date components
        data['Day_of_Week'] = data[self.datetime_col].dt.dayofweek
        data['Is_Weekend'] = data['Day_of_Week'].isin([5, 6]).astype(int)
        data['Month'] = data[self.datetime_col].dt.month
        data['Year'] = data[self.datetime_col].dt.year
        data['Day'] = data[self.datetime_col].dt.day
        
        # Add month indicators
        for month in range(1, 13):
            data[f'Is_Month_{month}'] = (data['Month'] == month).astype(int)
        
        # Add day of week indicators
        for day in range(7):
            data[f'Is_Day_{day}'] = (data['Day_of_Week'] == day).astype(int)
        
        # Add quarter information
        data['Quarter'] = data['Month'].apply(lambda x: (x-1)//3 + 1)
        for quarter in range(1, 5):
            data[f'Is_Q{quarter}'] = (data['Quarter'] == quarter).astype(int)
        
        logger.info(f"Added date features")
        return data
    
    def add_sku_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add SKU-specific features such as average price and quantity.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing SKU, price, and quantity columns.
        
        Returns
        -------
        pd.DataFrame
            Data with added SKU features.
        """
        if self.sku_col not in data.columns:
            logger.warning(f"SKU column {self.sku_col} not found, skipping SKU features")
            return data
            
        logger.info("Adding SKU features")
        
        # Add SKU counts
        sku_counts = data[self.sku_col].value_counts()
        data['SKU_Occurrence_Count'] = data[self.sku_col].map(sku_counts)
        
        # Add price features if price column exists
        if self.price_col in data.columns:
            # Calculate SKU price statistics
            sku_price_stats = data.groupby(self.sku_col)[self.price_col].agg(['mean', 'std', 'min', 'max'])
            sku_price_stats.columns = ['SKU_Mean_Price', 'SKU_Std_Price', 'SKU_Min_Price', 'SKU_Max_Price']
            
            # Add price coefficient of variation
            sku_price_stats['SKU_CV_Price'] = sku_price_stats['SKU_Std_Price'] / sku_price_stats['SKU_Mean_Price']
            
            # Merge price statistics back to data
            data = data.merge(sku_price_stats, left_on=self.sku_col, right_index=True, how='left')
            
            # Add price relative to SKU average
            data['Price_Relative_To_SKU_Avg'] = data[self.price_col] / data['SKU_Mean_Price']
        
        # Add product class features if available
        if self.product_class_col and self.product_class_col in data.columns:
            # Count SKUs per product class
            sku_per_class = data.groupby(self.product_class_col)[self.sku_col].nunique()
            data['SKUs_In_Class'] = data[self.product_class_col].map(sku_per_class)
            
            # Add product class statistics if price column exists
            if self.price_col in data.columns:
                class_price_stats = data.groupby(self.product_class_col)[self.price_col].agg(['mean', 'std'])
                class_price_stats.columns = ['Class_Mean_Price', 'Class_Std_Price']
                
                # Merge class price statistics back to data
                data = data.merge(class_price_stats, left_on=self.product_class_col, right_index=True, how='left')
                
                # Add price relative to class average
                data['Price_Relative_To_Class_Avg'] = data[self.price_col] / data['Class_Mean_Price']
        
        logger.info(f"Added SKU-related features")
        return data
    
    def apply_transforms(
        self, 
        data: pd.DataFrame,
        log_transform_price: bool = False,
        log_transform_quantity: bool = False
    ) -> pd.DataFrame:
        """
        Apply transformations to the data fields.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to transform
        log_transform_price : bool
            Whether to apply log transform to price
        log_transform_quantity : bool
            Whether to apply log transform to quantity
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        # Apply log transformations if requested
        if log_transform_price and self.price_col in data.columns:
            logger.info(f"Applying log transform to {self.price_col}")
            data[f"Log_{self.price_col}"] = np.log(data[self.price_col])
                
        if log_transform_quantity and self.quantity_col in data.columns:
            logger.info(f"Applying log transform to {self.quantity_col}")
            data[f"Log_{self.quantity_col}"] = np.log(data[self.quantity_col])
            
        return data 