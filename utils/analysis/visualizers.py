#!/usr/bin/env python3
"""
Visualization utilities for retail price elasticity analysis.

This module provides specialized visualization functions for analysis results,
focusing on creating clear, informative visualizations of elasticity data.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple

# Optional imports for visualization that might not be available
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("ArviZ not available. Some diagnostic features will be limited.")


def visualize_elasticities(elasticities: Dict[str, float], output_dir: str) -> None:
    """Create visualizations for elasticity results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert elasticities to DataFrame for easier plotting
    elasticity_df = pd.DataFrame.from_dict(
        elasticities, 
        orient='index', 
        columns=['elasticity']
    ).reset_index().rename(columns={'index': 'SKU'})
    
    # Create histogram of elasticities
    plt.figure(figsize=(10, 6))
    sns.histplot(elasticity_df['elasticity'], kde=True)
    plt.axvline(x=-1, color='r', linestyle='--')
    plt.title("Distribution of Price Elasticities")
    plt.xlabel("Elasticity")
    plt.ylabel("Count")
    plt.savefig(output_path / "elasticity_histogram.png")
    plt.close()
    
    # Create boxplot of elasticities
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=elasticity_df['elasticity'])
    plt.axhline(y=-1, color='r', linestyle='--')
    plt.title("Boxplot of Price Elasticities")
    plt.ylabel("Elasticity")
    plt.savefig(output_path / "elasticity_boxplot.png")
    plt.close()


def create_diagnostic_plots(trace: Any, output_dir: str) -> None:
    """Create diagnostic plots for MCMC sampling results."""
    if not ARVIZ_AVAILABLE:
        import logging
        logging.getLogger(__name__).warning("ArviZ not available. Cannot create diagnostic plots.")
        return
        
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot trace
    az.plot_trace(trace)
    plt.savefig(output_dir / "trace_plot.png")
    plt.close()
    
    # Plot posterior
    az.plot_posterior(trace)
    plt.savefig(output_dir / "posterior_plot.png")
    plt.close()
    
    # Plot rank plots
    az.plot_rank(trace)
    plt.savefig(output_dir / "rank_plot.png")
    plt.close()
    
    # Plot pair plot for key variables
    az.plot_pair(trace, var_names=["elasticity_mu", "class_effects"])
    plt.savefig(output_dir / "pair_plot.png")
    plt.close()


def create_model_comparison_plots(
    with_seasonality_elasticities: Dict[str, float],
    without_seasonality_elasticities: Dict[str, float],
    output_dir: str
) -> None:
    """Create visualizations comparing models with and without seasonality."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for sku in set(with_seasonality_elasticities.keys()) | set(without_seasonality_elasticities.keys()):
        comparison_data.append({
            'SKU': sku,
            'with_seasonality': with_seasonality_elasticities.get(sku, np.nan),
            'without_seasonality': without_seasonality_elasticities.get(sku, np.nan),
            'difference': with_seasonality_elasticities.get(sku, np.nan) - without_seasonality_elasticities.get(sku, np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path / "seasonality_comparison.csv", index=False)
    
    # Create comparison visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df['without_seasonality'], comparison_df['with_seasonality'], alpha=0.5)
    
    # Add identity line
    min_val = min(comparison_df['without_seasonality'].min(), comparison_df['with_seasonality'].min())
    max_val = max(comparison_df['without_seasonality'].max(), comparison_df['with_seasonality'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title("Elasticity Comparison: With vs. Without Seasonality")
    plt.xlabel("Elasticity Without Seasonality")
    plt.ylabel("Elasticity With Seasonality")
    plt.savefig(output_path / "seasonality_comparison.png")
    plt.close()
    
    # Create distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Create KDE plots
    sns.kdeplot(comparison_df['with_seasonality'].dropna(), 
               label='With Seasonality', fill=True, alpha=0.3)
    sns.kdeplot(comparison_df['without_seasonality'].dropna(), 
               label='Without Seasonality', fill=True, alpha=0.3)
    
    plt.axvline(x=-1, color='r', linestyle='--', label='Unit Elasticity')
    plt.title("Distribution of Elasticities: With vs. Without Seasonality")
    plt.xlabel("Elasticity")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path / "seasonality_distribution_comparison.png")
    plt.close() 