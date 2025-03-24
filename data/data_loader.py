#!/usr/bin/env python3
"""
Retail Data Loader Module for Price Elasticity Analysis.

This module provides a comprehensive DataLoader class designed specifically for handling retail sales data
needed for price elasticity modeling, with a focus on reliability and efficient preprocessing.

PURPOSE:
- Standardize the loading process for retail sales data across multiple formats (CSV, Parquet, Excel)
- Provide consistent preprocessing and data validation to ensure model compatibility
- Abstract data source complexity away from modeling components
- Enable reproducible data handling through consistent parameter settings

ASSUMPTIONS:
- Input data contains at minimum: SKU identifiers, prices, quantities, and product classes
- Dates are or can be converted to datetime format (if date features are used)
- Non-positive prices or quantities are treated as data errors and can be replaced
- Default column names follow a specific convention (can be overridden)
- Data can fit in memory for processing

EDGE CASES:
- Missing required columns trigger explicit DataLoaderError exceptions
- Column naming mismatches can be resolved using the column_mapping parameter
- Empty datasets are allowed but will trigger warnings
- Invalid date formats will be handled with conversion attempts and clear errors on failure
- Files with unsupported extensions will raise specific exceptions
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

from utils.logging_utils import logger
from utils.common import ensure_dir_exists, DataTransformer, validate_columns_exist

# Exception class for data loader errors
class DataLoaderError(Exception):
    """Exception raised for errors in the data loading process with details on the exact failure point."""
    pass


class DataLoader:
    """
    Specialized data loader for retail price elasticity analysis.
    
    This class handles all aspects of loading and preprocessing retail sales data for price elasticity modeling:
    - Loading from various file formats (CSV, Parquet, Excel)
    - Column validation and mapping to standardized names
    - Data filtering and sampling
    - Feature engineering for time series components
    - Optional log transformations for price and quantity
    - Statistical feature generation for SKU and product classes
    
    Key Benefits:
    - Data validation prevents common analysis errors
    - Flexible column mapping accommodates different data sources
    - Standardized preprocessing improves model consistency
    - Integrated logging provides transparency
    
    Limitations:
    - Processes entire datasets in memory
    - Does not support incremental loading for very large datasets
    - Limited categorical encoding options (currently just index-based)
    
    Parameters
    ----------
    data_path : str
        Path to the data file. Supports CSV, Parquet, and Excel formats.
    price_col : str
        Name of the column containing price information.
    quantity_col : str
        Name of the column containing quantity information.
    sku_col : str
        Name of the column containing SKU identifiers.
    product_class_col : str
        Name of the column containing product class information.
    date_col : str, optional
        Name of the column containing date information.
    """
    
    def __init__(
        self,
        data_path: str,
        price_col: str = "Price_Per_Unit",
        quantity_col: str = "Qty_Sold",
        sku_col: str = "SKU_Coded",
        product_class_col: str = "Product_Class_Code",
        date_col: Optional[str] = "Date"
    ):
        """Initialize the DataLoader with configuration parameters."""
        # Convert path to Path object
        self.data_path = Path(data_path)
        
        # Store column names
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.sku_col = sku_col
        self.product_class_col = product_class_col
        self.date_col = date_col
        
        # Define required columns
        self.required_columns = [
            price_col, 
            quantity_col, 
            sku_col, 
            product_class_col
        ]
        
        if date_col:
            self.required_columns.append(date_col)
        
        # Column mapping for flexibility with different naming conventions
        self.column_mapping = {}
        
        # Validate data path
        if not self.data_path.exists():
            raise DataLoaderError(f"Data file not found: {self.data_path}")
            
        # Check file format is supported
        if self.data_path.suffix.lower() not in ['.csv', '.parquet', '.xlsx', '.xls']:
            raise DataLoaderError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Initialize transformer
        self.transformer = DataTransformer()
        
        logger.info(f"Initialized DataLoader with data path: {self.data_path}")
        
    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set mapping from actual column names to expected column names.
        
        Parameters
        ----------
        mapping : Dict[str, str]
            Dictionary mapping actual column names to expected column names.
            Keys are the actual column names in the data,
            Values are the expected column names used by the loader.
            
        Example
        -------
        loader.set_column_mapping({
            'unit_price': 'Price_Per_Unit',
            'quantity': 'Qty_Sold',
            'product_id': 'SKU_Coded'
        })
        """
        self.column_mapping = mapping
        logger.info(f"Set column mapping: {mapping}")
        
    def _apply_column_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column mapping to the dataframe.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to apply the mapping to.
            
        Returns
        -------
        pd.DataFrame
            Data with columns renamed according to mapping.
        """
        if not self.column_mapping:
            return data
            
        # Create a column mapping dictionary (actual -> expected)
        rename_dict = {}
        for actual_col, expected_col in self.column_mapping.items():
            if actual_col in data.columns:
                rename_dict[actual_col] = expected_col
        
        if rename_dict:
            logger.info(f"Renaming columns: {rename_dict}")
            return data.rename(columns=rename_dict)
            
        return data
        
    def load_data(
        self,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        filter_conditions: Optional[Dict[str, Any]] = None,
        log_transform_price: bool = False,
        log_transform_quantity: bool = False,
        add_date_features: bool = True,
        add_sku_features: bool = True,
        preprocess: bool = True,
        min_observations_per_sku: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load and preprocess data from the specified source file.
        
        Parameters
        ----------
        sample_frac : float, optional
            Fraction of data to sample (0-1). If None, use all data.
        random_state : int, optional
            Random seed for reproducible sampling.
        filter_conditions : dict, optional
            Dictionary of column-value pairs to filter the data.
        log_transform_price : bool, optional
            If True, add log-transformed price column.
        log_transform_quantity : bool, optional
            If True, add log-transformed quantity column.
        add_date_features : bool, optional
            If True, extract and add date features (month, quarter, etc.).
        add_sku_features : bool, optional
            If True, add features related to SKUs.
        preprocess : bool, optional
            If True, perform basic preprocessing steps.
        min_observations_per_sku : int, optional
            Minimum number of observations to ensure per SKU.
            
        Returns
        -------
        pd.DataFrame
            Loaded and preprocessed data.
        """
        try:
            # Load data based on file type
            file_ext = self.data_path.suffix.lower()
            
            if file_ext == '.csv':
                data = pd.read_csv(self.data_path)
            elif file_ext == '.parquet':
                data = pd.read_parquet(self.data_path)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(self.data_path)
            else:
                raise DataLoaderError(f"Unsupported file format: {file_ext}")
            
            # Case-insensitive column handling
            case_insensitive_mapping = {}
            for col in data.columns:
                for req_col in self.required_columns:
                    if col.lower() == req_col.lower() and col != req_col:
                        case_insensitive_mapping[col] = req_col
                        logger.info(f"Found case-insensitive match: {col} -> {req_col}")
            
            if case_insensitive_mapping:
                data = data.rename(columns=case_insensitive_mapping)
                
            # Apply column mapping if set
            if self.column_mapping:
                data = self._apply_column_mapping(data)
            
            # Try to infer column mappings if possible
            self._attempt_column_inference(data)
                
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                raise DataLoaderError(f"Missing required columns: {missing_cols}")
                
            # Apply preprocessing if requested
            if preprocess:
                # Remove rows with missing values in key columns
                data = data.dropna(subset=self.required_columns)
                
                # Apply filters if specified
                if filter_conditions:
                    for col, value in filter_conditions.items():
                        if col in data.columns:
                            if isinstance(value, (list, tuple)):
                                data = data[data[col].isin(value)]
                            else:
                                data = data[data[col] == value]
                        else:
                            logger.warning(f"Filter column not found: {col}")
                
                # Add date features if requested and date column is available
                if add_date_features and self.date_col and self.date_col in data.columns:
                    data = self.transformer.add_date_features(data, self.date_col)
                
                # Add log-transformed columns
                if log_transform_price:
                    data = self.transformer.log_transform(data, self.price_col)
                
                if log_transform_quantity:
                    data = self.transformer.log_transform(data, self.quantity_col)
                
                # Add SKU-level features if requested
                if add_sku_features:
                    # Calculate price stats by SKU
                    data = self.transformer.add_sku_stats(data, self.sku_col, self.price_col)
                    
                    # Calculate quantity stats by SKU
                    data = self.transformer.add_sku_stats(data, self.sku_col, self.quantity_col)
                    
                    # Calculate product class stats for price
                    if validate_columns_exist(data, [self.product_class_col, self.price_col]):
                        class_price_stats = data.groupby(self.product_class_col)[self.price_col].agg(['mean', 'std']).reset_index()
                        class_price_stats.columns = [self.product_class_col, f"{self.price_col}_class_mean", f"{self.price_col}_class_std"]
                        data = pd.merge(data, class_price_stats, on=self.product_class_col, how='left')
            
            # Sample data if requested
            if sample_frac is not None and 0 < sample_frac < 1:
                if min_observations_per_sku is not None:
                    data = self._stratified_sample(data, sample_frac, min_observations_per_sku)
                else:
                    data = data.sample(frac=sample_frac, random_state=random_state)
                logger.info(f"Created sampled data with {len(data)} rows")
            
            logger.info(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
            return data
            
        except Exception as e:
            # Re-raise with more context
            error_msg = f"Error loading data from {self.data_path}: {str(e)}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg) from e
            
    def _attempt_column_inference(self, data: pd.DataFrame) -> None:
        """
        Attempt to infer column mappings from data.
        
        This method tries to guess which columns correspond to 
        price, quantity, SKU, etc. based on naming patterns.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to infer columns from.
        """
        # Skip if all required columns are present
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if not missing_cols:
            return
            
        logger.info(f"Attempting to infer missing columns: {missing_cols}")
        
        # Common column name patterns
        price_patterns = ['price', 'unit_price', 'cost', 'amount']
        quantity_patterns = ['qty', 'quantity', 'volume', 'units', 'sales']
        sku_patterns = ['sku', 'item', 'product', 'id', 'code']
        date_patterns = ['date', 'day', 'period', 'time']
        
        # Mapping from pattern types to expected column names
        pattern_mapping = {
            'price': self.price_col,
            'quantity': self.quantity_col,
            'sku': self.sku_col,
            'date': self.date_col
        }
        
        # Try to find matching columns
        inferred_mapping = {}
        
        # Check each column in the data
        for col in data.columns:
            col_lower = col.lower()
            
            # Try to match with patterns
            if self.price_col in missing_cols and any(p in col_lower for p in price_patterns):
                inferred_mapping[col] = self.price_col
            elif self.quantity_col in missing_cols and any(p in col_lower for p in quantity_patterns):
                inferred_mapping[col] = self.quantity_col
            elif self.sku_col in missing_cols and any(p in col_lower for p in sku_patterns):
                inferred_mapping[col] = self.sku_col
            elif self.date_col in missing_cols and any(p in col_lower for p in date_patterns):
                inferred_mapping[col] = self.date_col
        
        # Update column mapping if inferences were made
        if inferred_mapping:
            logger.info(f"Inferred column mapping: {inferred_mapping}")
            self.column_mapping.update(inferred_mapping)
    
    def save_processed_data(self, data: pd.DataFrame, output_path: Union[str, Path], 
                           format: str = 'parquet') -> None:
        """
        Save processed data to a file.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to save.
        output_path : str or Path
            The path to save the data to.
        format : str
            The file format to use ('csv', 'parquet', or 'excel').
        """
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        ensure_dir_exists(output_path.parent)
        
        try:
            if format.lower() == 'csv':
                data.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                data.to_parquet(output_path, index=False)
            elif format.lower() == 'excel':
                data.to_excel(output_path, index=False)
            else:
                raise DataLoaderError(f"Unsupported output format: {format}")
                
            logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            error_msg = f"Error saving data to {output_path}: {str(e)}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg) from e
    
    @staticmethod
    def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to get information about.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with data information.
        """
        info = {
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'column_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        }
        
        # Add column unique values for categorical columns
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if data[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                info[f'{col}_unique_values'] = data[col].value_counts().to_dict()
                
        return info

    def _stratified_sample(self, df, sample_frac, min_obs_per_sku):
        """
        Perform stratified sampling to ensure minimum observations per SKU.
        
        Args:
            df: Input DataFrame
            sample_frac: Overall sampling fraction
            min_obs_per_sku: Minimum observations to ensure per SKU
            
        Returns:
            Sampled DataFrame with minimum observations per SKU
        """
        result = pd.DataFrame()
        sku_col = self.sku_col
        
        # Get all unique SKUs
        skus = df[sku_col].unique()
        
        for sku in skus:
            sku_data = df[df[sku_col] == sku]
            
            if len(sku_data) <= min_obs_per_sku:
                # Keep all rows if fewer than minimum
                sampled_sku = sku_data
            else:
                # Calculate how many to sample, but at least min_obs_per_sku
                n_to_sample = max(int(len(sku_data) * sample_frac), min_obs_per_sku)
                sampled_sku = sku_data.sample(n=n_to_sample, random_state=42)
            
            result = pd.concat([result, sampled_sku])
            
        logger.info(f"Stratified sampling: {len(result)} rows with minimum {min_obs_per_sku} obs per SKU")
        return result


if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader(
        data_path="path/to/your/data.parquet",
        price_col="Price_Per_Unit",
        quantity_col="Qty_Sold",
        sku_col="SKU_Coded",
        product_class_col="Product_Class_Code",
        date_col="Date"
    )
    
    # Load and analyze data
    data = data_loader.load_data(sample_frac=0.1)
    
    logger.info(f"Loaded data with shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}") 