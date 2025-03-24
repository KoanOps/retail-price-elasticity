#!/usr/bin/env python3
"""
Data Visualizer for Retail Price Elasticity Analysis

This module provides visualization and diagnostic functionality for retail data analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path

# Use absolute imports
from utils.logging_utils import logger
from utils.common import ensure_dir_exists


class DataVisualizer:
    """
    Handles data visualization and diagnostics for retail price elasticity analysis.
    
    This class is responsible for:
    - Generating diagnostic visualizations
    - Creating summary statistics
    - Visualizing price-quantity relationships
    - Analyzing SKU and product class distributions
    
    Parameters
    ----------
    results_dir : str or Path
        Directory to save results and visualizations.
    price_col : str
        Name of the column containing price information.
    quantity_col : str
        Name of the column containing quantity information.
    sku_col : str
        Name of the column containing SKU identifiers.
    product_class_col : str, optional
        Name of the column containing product class information.
    datetime_col : str, optional
        Name of the column containing datetime information.
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        price_col: str = "Price_Per_Unit",
        quantity_col: str = "Qty_Sold",
        sku_col: str = "SKU_Coded",
        product_class_col: Optional[str] = "Product_Class_Code",
        datetime_col: Optional[str] = "Date"
    ):
        self.results_dir = Path(results_dir)
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.sku_col = sku_col
        self.product_class_col = product_class_col
        self.datetime_col = datetime_col
        
        # Create results directory if it doesn't exist
        ensure_dir_exists(self.results_dir)
    
    def generate_data_diagnostics(self, data: pd.DataFrame, target_dir: Optional[str] = None) -> None:
        """
        Generate diagnostic visualizations for the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to generate diagnostics for.
        target_dir : str or Path, optional
            Directory to save the diagnostic visualizations.
            If None, uses the results_dir specified at initialization.
        """
        logger.info("Generating data diagnostics")
        
        # Set up the target directory
        if target_dir is None:
            target_dir = self.results_dir / "data_diagnostics"
        else:
            target_dir = Path(target_dir)
        
        ensure_dir_exists(target_dir)
        
        # Create summary statistics report
        try:
            summary_stats = data.describe().transpose()
            summary_stats.to_csv(target_dir / "summary_statistics.csv")
            logger.info(f"Saved summary statistics")
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
        
        # Save column data types
        try:
            pd.DataFrame({"Column": data.columns, "Data_Type": data.dtypes}).to_csv(
                target_dir / "column_types.csv", index=False
            )
            logger.info(f"Saved column types")
        except Exception as e:
            logger.error(f"Error saving column types: {e}")
        
        # Generate price-quantity visualizations
        self._generate_price_quantity_plots(data, target_dir)
        
        # Generate SKU visualizations
        self._generate_sku_plots(data, target_dir)
        
        # Generate product class visualizations
        self._generate_product_class_plots(data, target_dir)
        
        # Generate time visualizations
        self._generate_time_plots(data, target_dir)
        
        logger.info("Data diagnostics generation complete")
    
    def _generate_price_quantity_plots(self, data: pd.DataFrame, target_dir: Path) -> None:
        """Generate price and quantity related visualizations."""
        if self.price_col in data.columns and self.quantity_col in data.columns:
            try:
                # Price distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(x=data[self.price_col], kde=True)
                plt.title(f"Distribution of {self.price_col}")
                plt.xlabel(self.price_col)
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(target_dir / "price_distribution.png", dpi=300)
                plt.close()
                
                # Quantity distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(x=data[self.quantity_col], kde=True)
                plt.title(f"Distribution of {self.quantity_col}")
                plt.xlabel(self.quantity_col)
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(target_dir / "quantity_distribution.png", dpi=300)
                plt.close()
                
                # Log transformations if values are positive
                if (data[self.price_col] > 0).all():
                    plt.figure(figsize=(10, 6))
                    sns.histplot(x=np.log(data[self.price_col]), kde=True)
                    plt.title(f"Distribution of Log({self.price_col})")
                    plt.xlabel(f"Log({self.price_col})")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    plt.savefig(target_dir / "log_price_distribution.png", dpi=300)
                    plt.close()
                
                if (data[self.quantity_col] > 0).all():
                    plt.figure(figsize=(10, 6))
                    sns.histplot(x=np.log(data[self.quantity_col]), kde=True)
                    plt.title(f"Distribution of Log({self.quantity_col})")
                    plt.xlabel(f"Log({self.quantity_col})")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    plt.savefig(target_dir / "log_quantity_distribution.png", dpi=300)
                    plt.close()
                
                # Price vs Quantity scatter plot
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    x=self.price_col, 
                    y=self.quantity_col, 
                    data=data.sample(min(1000, len(data)), random_state=42),
                    alpha=0.5
                )
                plt.title(f"{self.price_col} vs {self.quantity_col}")
                plt.xlabel(self.price_col)
                plt.ylabel(self.quantity_col)
                plt.tight_layout()
                plt.savefig(target_dir / "price_vs_quantity.png", dpi=300)
                plt.close()
                
                # Log-Log Price vs Quantity (for elasticity visualization)
                if (data[self.price_col] > 0).all() and (data[self.quantity_col] > 0).all():
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        x=np.log(data[self.price_col]), 
                        y=np.log(data[self.quantity_col]), 
                        data=data.sample(min(1000, len(data)), random_state=42),
                        alpha=0.5
                    )
                    plt.title(f"Log({self.price_col}) vs Log({self.quantity_col})")
                    plt.xlabel(f"Log({self.price_col})")
                    plt.ylabel(f"Log({self.quantity_col})")
                    plt.tight_layout()
                    plt.savefig(target_dir / "log_price_vs_log_quantity.png", dpi=300)
                    plt.close()
                
                logger.info(f"Generated price-quantity visualizations")
            except Exception as e:
                logger.error(f"Error generating price-quantity visualizations: {e}")
    
    def _generate_sku_plots(self, data: pd.DataFrame, target_dir: Path) -> None:
        """Generate SKU-related visualizations."""
        if self.sku_col in data.columns:
            try:
                # SKU count bar chart (top 20)
                plt.figure(figsize=(12, 8))
                sku_counts = data[self.sku_col].value_counts().head(20)
                sns.barplot(x=sku_counts.index, y=sku_counts.values)
                plt.title("Top 20 SKUs by Frequency")
                plt.xlabel("SKU")
                plt.ylabel("Count")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(target_dir / "top_skus.png", dpi=300)
                plt.close()
                
                # SKU distribution summary
                sku_distribution = pd.DataFrame({
                    "Metric": ["Total SKUs", "Min SKU Count", "Max SKU Count", "Mean SKU Count", "Median SKU Count"],
                    "Value": [
                        data[self.sku_col].nunique(),
                        data[self.sku_col].value_counts().min(),
                        data[self.sku_col].value_counts().max(),
                        data[self.sku_col].value_counts().mean(),
                        data[self.sku_col].value_counts().median()
                    ]
                })
                sku_distribution.to_csv(target_dir / "sku_distribution.csv", index=False)
                
                logger.info(f"Generated SKU diagnostics")
            except Exception as e:
                logger.error(f"Error generating SKU diagnostics: {e}")
    
    def _generate_product_class_plots(self, data: pd.DataFrame, target_dir: Path) -> None:
        """Generate product class visualizations."""
        if self.product_class_col and self.product_class_col in data.columns:
            try:
                # Product class count bar chart
                plt.figure(figsize=(12, 8))
                class_counts = data[self.product_class_col].value_counts()
                sns.barplot(x=class_counts.index, y=class_counts.values)
                plt.title("Product Classes by Frequency")
                plt.xlabel("Product Class")
                plt.ylabel("Count")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(target_dir / "product_classes.png", dpi=300)
                plt.close()
                
                # SKUs per product class
                plt.figure(figsize=(12, 8))
                skus_per_class = data.groupby(self.product_class_col)[self.sku_col].nunique().sort_values(ascending=False)
                sns.barplot(x=skus_per_class.index, y=skus_per_class.values)
                plt.title("Number of Unique SKUs per Product Class")
                plt.xlabel("Product Class")
                plt.ylabel("Number of SKUs")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(target_dir / "skus_per_class.png", dpi=300)
                plt.close()
                
                logger.info(f"Generated product class diagnostics")
            except Exception as e:
                logger.error(f"Error generating product class diagnostics: {e}")
    
    def _generate_time_plots(self, data: pd.DataFrame, target_dir: Path) -> None:
        """Generate time-based visualizations."""
        if self.datetime_col and self.datetime_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[self.datetime_col]):
            try:
                # Time range summary
                time_summary = pd.DataFrame({
                    "Metric": ["Start Date", "End Date", "Date Range (days)", "Number of Unique Dates"],
                    "Value": [
                        data[self.datetime_col].min().strftime('%Y-%m-%d'),
                        data[self.datetime_col].max().strftime('%Y-%m-%d'),
                        (data[self.datetime_col].max() - data[self.datetime_col].min()).days,
                        data[self.datetime_col].dt.date.nunique()
                    ]
                })
                time_summary.to_csv(target_dir / "time_summary.csv", index=False)
                
                # Transactions by month
                plt.figure(figsize=(12, 6))
                monthly_counts = data.groupby(data[self.datetime_col].dt.to_period('M')).size()
                monthly_counts.index = monthly_counts.index.astype(str)
                sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
                plt.title("Transactions by Month")
                plt.xlabel("Month")
                plt.ylabel("Count")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(target_dir / "transactions_by_month.png", dpi=300)
                plt.close()
                
                # Transactions by day of week
                plt.figure(figsize=(10, 6))
                day_of_week_counts = data.groupby(data[self.datetime_col].dt.dayofweek).size()
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Create a new Series with day names as index
                labeled_counts = pd.Series(
                    day_of_week_counts.values,
                    index=[day_names[i] for i in day_of_week_counts.index]
                )
                
                sns.barplot(x=labeled_counts.index, y=labeled_counts.values)
                plt.title("Transactions by Day of Week")
                plt.xlabel("Day of Week")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(target_dir / "transactions_by_day.png", dpi=300)
                plt.close()
                
                logger.info(f"Generated date diagnostics")
            except Exception as e:
                logger.error(f"Error generating date diagnostics: {e}")
                
    def export_data_summary(self, data: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """
        Export a detailed summary of the data to a CSV file.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to summarize.
        file_path : str or Path, optional
            Path to save the summary CSV file.
            If None, uses the results_dir with a default filename.
        """
        if file_path is None:
            file_path = self.results_dir / "data_summary.csv"
        else:
            file_path = Path(file_path)
        
        # Ensure directory exists
        ensure_dir_exists(file_path.parent)
        
        try:
            # Basic column info
            column_info = []
            for col in data.columns:
                col_type = data[col].dtype
                non_null = data[col].count()
                null_count = data[col].isna().sum()
                null_pct = null_count / len(data) * 100
                
                if pd.api.types.is_numeric_dtype(data[col]):
                    min_val = data[col].min()
                    max_val = data[col].max()
                    mean_val = data[col].mean()
                    median_val = data[col].median()
                    std_val = data[col].std()
                    unique_vals = data[col].nunique()
                else:
                    min_val = "N/A"
                    max_val = "N/A"
                    mean_val = "N/A"
                    median_val = "N/A"
                    std_val = "N/A"
                    unique_vals = data[col].nunique()
                
                column_info.append({
                    "Column": col,
                    "Type": col_type,
                    "Non-Null Count": non_null,
                    "Null Count": null_count,
                    "Null %": null_pct,
                    "Unique Values": unique_vals,
                    "Min": min_val,
                    "Max": max_val,
                    "Mean": mean_val,
                    "Median": median_val,
                    "Std Dev": std_val
                })
            
            summary_df = pd.DataFrame(column_info)
            summary_df.to_csv(file_path, index=False)
            logger.info(f"Data summary exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data summary: {e}") 