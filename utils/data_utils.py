#!/usr/bin/env python3
"""
Data utility functions for the Retail package.

This module provides data transformation and validation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from utils.logging_utils import logger


class DataTransformer:
    """
    A utility class for common data transformations.
    This centralizes transformation logic to avoid duplication.
    """
    
    @staticmethod
    def log_transform(df: pd.DataFrame, column: str, add_column: bool = True) -> pd.DataFrame:
        """
        Apply log transformation to a column.
        
        Args:
            df: DataFrame containing the column
            column: Name of column to transform
            add_column: If True, add a new column with '_log' suffix
                       otherwise, replace the original column
                       
        Returns:
            DataFrame with transformed data
        """
        if not validate_column_exists(df, column):
            return df
            
        # Create a copy to avoid modifying the input
        result = df.copy()
        
        # Apply log transformation, handling zeros and negative values
        transformed = np.log(np.maximum(result[column], 0.0001))
        
        if add_column:
            result[f"log_{column}"] = transformed
        else:
            result[column] = transformed
            
        return result
        
    @staticmethod
    def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Add date-related features to a DataFrame.
        
        Args:
            df: DataFrame containing the date column
            date_col: Name of date column
            
        Returns:
            DataFrame with added date features
        """
        if not validate_column_exists(df, date_col):
            return df
            
        # Create a copy to avoid modifying the input
        result = df.copy()
        
        # Ensure date column is datetime type
        result[date_col] = pd.to_datetime(result[date_col])
        
        # Extract useful date components
        result['day_of_week'] = result[date_col].dt.dayofweek
        result['month'] = result[date_col].dt.month
        result['quarter'] = result[date_col].dt.quarter
        result['year'] = result[date_col].dt.year
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        
        return result
        
    @staticmethod
    def add_sku_stats(df: pd.DataFrame, sku_col: str, value_col: str) -> pd.DataFrame:
        """
        Add SKU-level statistics.
        
        Args:
            df: DataFrame containing SKU and value columns
            sku_col: Name of SKU column
            value_col: Name of value column to compute stats on
            
        Returns:
            DataFrame with added SKU stats
        """
        if not validate_columns_exist(df, [sku_col, value_col]):
            return df
            
        # Create a copy to avoid modifying the input
        result = df.copy()
        
        # Compute SKU-level statistics
        sku_stats = df.groupby(sku_col)[value_col].agg(['mean', 'std', 'min', 'max']).reset_index()
        sku_stats.columns = [sku_col, f'{value_col}_mean', f'{value_col}_std', 
                           f'{value_col}_min', f'{value_col}_max']
        
        # Merge with original data
        result = pd.merge(result, sku_stats, on=sku_col, how='left')
        
        return result
        
    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize a column using various methods.
        
        Args:
            df: DataFrame containing the column
            column: Name of column to normalize
            method: Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
            DataFrame with normalized column
        """
        if not validate_column_exists(df, column):
            return df
            
        # Create a copy to avoid modifying the input
        result = df.copy()
        
        if method == 'zscore':
            # Z-score normalization
            mean = result[column].mean()
            std = result[column].std()
            if std > 0:
                result[f'{column}_norm'] = (result[column] - mean) / std
            else:
                result[f'{column}_norm'] = 0
                
        elif method == 'minmax':
            # Min-max normalization
            min_val = result[column].min()
            max_val = result[column].max()
            if max_val > min_val:
                result[f'{column}_norm'] = (result[column] - min_val) / (max_val - min_val)
            else:
                result[f'{column}_norm'] = 0
                
        elif method == 'robust':
            # Robust normalization using median and IQR
            median = result[column].median()
            q1 = result[column].quantile(0.25)
            q3 = result[column].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                result[f'{column}_norm'] = (result[column] - median) / iqr
            else:
                result[f'{column}_norm'] = 0
                
        return result


def validate_column_exists(df: pd.DataFrame, column: str, description: str = "DataFrame") -> bool:
    """
    Validate that a column exists in a DataFrame.
    
    Args:
        df: DataFrame to check
        column: Column name to check for
        description: Description for error message
        
    Returns:
        True if column exists, False otherwise
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in {description}")
        return False
    return True


def validate_columns_exist(df: pd.DataFrame, columns: List[str], description: str = "DataFrame") -> bool:
    """
    Validate that all columns exist in a DataFrame.
    
    Args:
        df: DataFrame to check
        columns: List of column names to check for
        description: Description for error message
        
    Returns:
        True if all columns exist, False otherwise
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Columns {missing_columns} not found in {description}")
        return False
    return True


def calculate_summary_stats(series: pd.Series) -> Dict[str, float]:
    """
    Calculate summary statistics for a Series.
    
    Args:
        series: Series to calculate statistics for
        
    Returns:
        Dictionary of summary statistics
    """
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75)
    }


def split_train_test(
    df: pd.DataFrame, 
    date_column: str,
    split_date: str,
    test_size: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and test sets.
    
    Args:
        df: DataFrame to split
        date_column: Name of date column
        split_date: Date to split on (before: train, after: test)
        test_size: Fraction of data to use for testing (if split_date not provided)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if not validate_column_exists(df, date_column):
        # If validation fails, return empty DataFrames
        return df.head(0), df.head(0)
        
    # Ensure date column is datetime type
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    if split_date:
        # Split by date
        split_date_dt = pd.to_datetime(split_date)
        train_df = df[df[date_column] < split_date_dt]
        test_df = df[df[date_column] >= split_date_dt]
        
        logger.info(f"Split data by date: {len(train_df)} train rows, {len(test_df)} test rows")
        
    elif test_size:
        # Split by fraction
        df = df.sort_values(date_column)  # Sort by date
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Split data by fraction: {len(train_df)} train rows, {len(test_df)} test rows")
        
    else:
        # Default: use 80% for training
        df = df.sort_values(date_column)  # Sort by date
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Split data with default 80/20: {len(train_df)} train rows, {len(test_df)} test rows")
    
    return train_df, test_df


def batch_process(items: List[Any], 
                  process_func: Callable, 
                  batch_size: int = 100, 
                  **kwargs) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each batch of items
        batch_size: Size of each batch
        **kwargs: Additional arguments to pass to process_func
        
    Returns:
        List of processed items
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch, **kwargs)
        results.extend(batch_results)
        logger.debug(f"Processed batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
    
    return results 