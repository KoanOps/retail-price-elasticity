#!/usr/bin/env python
"""
Log Transform Validation Utility

This script provides tools to evaluate whether log transformations are appropriate
for retail data variables based on statistical tests and visual diagnostics.

PURPOSE:
- Validate if price, quantity, or sales variables follow log-normal distributions
- Test custom data distributions for log-normality
- Generate diagnostic plots to visualize transformation effects
- Provide objective metrics for transformation decisions

USAGE:
    # Check log transform appropriateness for data file:
    python utils/analysis/log_transform_validator.py --data-path data/sales.parquet
    
    # Run test suite with synthetic data:
    python utils/analysis/log_transform_validator.py --test-synthetic
    
    # Check specific columns in a data file:
    python utils/analysis/log_transform_validator.py --data-path data/sales.parquet --columns price quantity

VALIDATION APPROACH:
The validation uses three key metrics to evaluate log transformation appropriateness:
1. Skewness comparison between raw and log-transformed data
2. Shapiro-Wilk normality test p-values
3. Q-Q plot correlation coefficients

A log transformation is deemed appropriate if at least 2 of these 3 metrics
improve after transformation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.data_preparation import BayesianDataPreparation
from utils.logging_utils import logger

def create_diagnostics_dir(diagnostics_dir: str = "diagnostics") -> str:
    """Create diagnostics directory if it doesn't exist."""
    os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_dir

def check_real_data(
    data_path: str,
    columns: Optional[List[str]] = None,
    diagnostics_dir: str = "diagnostics"
) -> Dict[str, bool]:
    """
    Check if log transformation is appropriate for columns in a real dataset.
    
    Args:
        data_path: Path to the data file (parquet, csv)
        columns: List of column names to check (if None, will try common retail columns)
        diagnostics_dir: Directory to save diagnostic plots
        
    Returns:
        Dictionary mapping column names to log transform appropriateness (True/False)
    """
    # Load data based on file extension
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext == '.parquet':
        data = pd.read_parquet(data_path)
    elif file_ext == '.csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    logger.info(f'Loaded data from {data_path} with {len(data)} rows and columns: {list(data.columns)}')
    
    # Create Price_Per_Unit if it doesn't exist but Total_Sale_Value and Qty_Sold do
    if 'Price_Per_Unit' not in data.columns and 'Total_Sale_Value' in data.columns and 'Qty_Sold' in data.columns:
        logger.info("Calculating Price_Per_Unit from Total_Sale_Value / Qty_Sold")
        data['Price_Per_Unit'] = data['Total_Sale_Value'] / data['Qty_Sold']
        # Replace infinite values with NaN
        data['Price_Per_Unit'].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop NaN values for analysis
        data = data.dropna(subset=['Price_Per_Unit'])
        logger.info(f"Created Price_Per_Unit column with {len(data)} valid rows")
    
    # Define columns to check if not specified
    if columns is None:
        # Try common column names in retail datasets
        potential_columns = [
            # Price columns
            'Price_Per_Unit', 'price', 'unit_price', 'Price', 'UnitPrice',
            # Quantity columns
            'Qty_Sold', 'quantity', 'Quantity', 'Volume', 'Sales_Units',
            # Sales value columns
            'Total_Sale_Value', 'sales', 'Sales', 'Revenue', 'Amount'
        ]
        columns = [col for col in potential_columns if col in data.columns]
    else:
        # Filter to columns that exist in the data
        columns = [col for col in columns if col in data.columns]
    
    if not columns:
        logger.warning("No valid columns found to check for log transformation")
        return {}
    
    # Create data preparation object with diagnostic plots enabled
    data_prep = BayesianDataPreparation(
        data_config={'save_diagnostic_plots': True, 'diagnostics_dir': diagnostics_dir}
    )
    
    # Check each column
    results = {}
    for column in columns:
        logger.info(f'\n=== TESTING {column} COLUMN ===')
        # Skip columns with non-positive values
        if (data[column] <= 0).any():
            non_positive_count = (data[column] <= 0).sum()
            logger.warning(f"Column {column} has {non_positive_count} non-positive values")
            logger.info(f"Testing will proceed, but non-positive values will be temporarily replaced")
        
        # Run validation
        is_appropriate = data_prep._validate_log_transformation(data[column], column)
        results[column] = is_appropriate
        
        # Log summary statistics
        logger.info(f'Should log transform {column}? {is_appropriate}')
        logger.info(f'{column} summary stats: \n{data[column].describe()}')
    
    return results

def test_synthetic_distributions(diagnostics_dir: str = "diagnostics") -> Dict[str, bool]:
    """
    Test log transformation appropriateness on synthetic data distributions.
    
    This is useful for validating the algorithm against known distributions.
    
    Args:
        diagnostics_dir: Directory to save diagnostic plots
        
    Returns:
        Dictionary mapping distribution names to log transform appropriateness (True/False)
    """
    # Create sample data with different distributions
    np.random.seed(42)  # For reproducibility
    
    # Create log-normal data (should benefit from log transform)
    lognormal_data = np.random.lognormal(mean=0, sigma=1, size=1000)
    # Create normal data (should not benefit from log transform)
    normal_data = np.random.normal(loc=5, scale=1, size=1000)
    # Create right-skewed data that's not quite lognormal
    skewed_data = np.random.gamma(2, 2, size=1000)
    
    # Initialize data preparation
    data_prep = BayesianDataPreparation(
        data_config={"save_diagnostic_plots": True, "diagnostics_dir": diagnostics_dir}
    )
    
    # Test each distribution
    results = {}
    
    logger.info("\n=== TESTING LOG-NORMAL DATA ===")
    lognormal_series = pd.Series(lognormal_data)
    lognormal_result = data_prep._validate_log_transformation(lognormal_series, "lognormal")
    logger.info(f"Should log transform log-normal data? {lognormal_result}")
    results["lognormal"] = lognormal_result
    
    logger.info("\n=== TESTING NORMAL DATA ===")
    normal_series = pd.Series(normal_data)
    normal_result = data_prep._validate_log_transformation(normal_series, "normal")
    logger.info(f"Should log transform normal data? {normal_result}")
    results["normal"] = normal_result
    
    logger.info("\n=== TESTING SKEWED DATA ===")
    skewed_series = pd.Series(skewed_data)
    skewed_result = data_prep._validate_log_transformation(skewed_series, "skewed")
    logger.info(f"Should log transform skewed data? {skewed_result}")
    results["skewed"] = skewed_result
    
    # Plot distributions before and after transformation
    plt.figure(figsize=(15, 10))
    
    # Plot lognormal data
    plt.subplot(3, 2, 1)
    sns.histplot(lognormal_data, kde=True)
    plt.title('Log-normal data (original)')
    
    plt.subplot(3, 2, 2)
    sns.histplot(np.log(lognormal_data), kde=True)
    plt.title('Log-normal data (log-transformed)')
    
    # Plot normal data
    plt.subplot(3, 2, 3)
    sns.histplot(normal_data, kde=True)
    plt.title('Normal data (original)')
    
    plt.subplot(3, 2, 4)
    sns.histplot(np.log(normal_data), kde=True)
    plt.title('Normal data (log-transformed)')
    
    # Plot skewed data
    plt.subplot(3, 2, 5)
    sns.histplot(skewed_data, kde=True)
    plt.title('Skewed data (original)')
    
    plt.subplot(3, 2, 6)
    sns.histplot(np.log(skewed_data), kde=True)
    plt.title('Skewed data (log-transformed)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, 'synthetic_log_transform_comparison.png'))
    plt.close()
    
    logger.info("\nSynthetic test complete. Results:")
    for dist, is_appropriate in results.items():
        logger.info(f"- {dist}: {'Appropriate' if is_appropriate else 'Not appropriate'} for log transform")
    
    return results

def explain_validation_method():
    """Print an explanation of the log transform validation method."""
    explanation = """
Explanation of the Log Transform Validation Method:
--------------------------------------------------
The algorithm evaluates three key metrics to determine if log transformation is appropriate:

1. Skewness: Compares if log transformation reduces skewness toward zero
   - Raw skewness measures asymmetry in the original distribution
   - Log-transformed skewness measures asymmetry after applying log
   - A reduction in absolute skewness suggests improvement

2. Shapiro-Wilk test: Checks if log transformation improves normality
   - Higher p-value indicates more normal-like distribution
   - Very small p-values (<0.05) suggest non-normality
   - If log transform has higher p-value, it improved normality

3. Q-Q plot correlation: Measures linearity in quantile-quantile plot
   - Correlation close to 1.0 indicates normal distribution
   - Higher correlation after log transform suggests improvement
   - This is a robust visual confirmation of normality

A log transformation is deemed appropriate if at least 2 of these 3 metrics improve after transformation.
This approach helps identify variables that follow a log-normal distribution or are right-skewed.

Common applications in retail include:
- Price data (typically right-skewed)
- Quantity sold (often follows power law distribution)
- Sales values (combination of price and quantity effects)
"""
    print(explanation)
    return explanation

def main():
    """Parse arguments and run the appropriate validation."""
    parser = argparse.ArgumentParser(description="Validate log transformation appropriateness for data")
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file (parquet or csv)"
    )
    
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Specific columns to check (space-separated)"
    )
    
    parser.add_argument(
        "--test-synthetic",
        action="store_true",
        help="Run tests on synthetic distributions"
    )
    
    parser.add_argument(
        "--diagnostics-dir",
        default="diagnostics",
        help="Directory to save diagnostic plots"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Explain the validation method"
    )
    
    args = parser.parse_args()
    
    # Create diagnostics directory
    diagnostics_dir = create_diagnostics_dir(args.diagnostics_dir)
    
    # Show explanation if requested
    if args.explain:
        explain_validation_method()
        if not args.data_path and not args.test_synthetic:
            return
    
    # Run requested analysis
    if args.test_synthetic:
        logger.info("Running tests on synthetic distributions")
        test_synthetic_distributions(diagnostics_dir)
    
    if args.data_path:
        logger.info(f"Checking log transformation appropriateness for {args.data_path}")
        check_real_data(args.data_path, args.columns, diagnostics_dir)
    
    # If no actions specified, show help
    if not args.data_path and not args.test_synthetic and not args.explain:
        parser.print_help()

if __name__ == "__main__":
    main() 