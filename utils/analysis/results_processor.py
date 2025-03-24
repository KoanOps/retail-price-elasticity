#!/usr/bin/env python3
"""
Results processing utilities for retail price elasticity analysis.

This module provides functions for processing and saving retail price elasticity
analysis results in various formats for further use or reporting.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple

from utils.logging_utils import logger


def save_elasticity_results(elasticities: Dict[str, float], output_dir: str) -> Dict[str, Any]:
    """Save elasticity results to disk and return summary statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for easier processing
    elasticity_df = pd.DataFrame.from_dict(
        elasticities, 
        orient='index', 
        columns=['elasticity']
    ).reset_index().rename(columns={'index': 'SKU'})
    
    # Save to CSV
    elasticity_df.to_csv(output_path / "elasticities.csv", index=False)
    
    # Calculate summary statistics
    summary = {
        'mean_elasticity': float(elasticity_df['elasticity'].mean()),
        'median_elasticity': float(elasticity_df['elasticity'].median()),
        'min_elasticity': float(elasticity_df['elasticity'].min()),
        'max_elasticity': float(elasticity_df['elasticity'].max()),
        'std_elasticity': float(elasticity_df['elasticity'].std()),
        'sample_size': len(elasticity_df)
    }
    
    # Save summary statistics
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Saved elasticity results to {output_path}")
    return summary


def save_model_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save full model results to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save elasticity results if available
    if 'elasticities' in results:
        save_elasticity_results(results['elasticities'], output_path)
    
    # Save other serializable result components
    serializable_results = {}
    for key, value in results.items():
        # Skip non-serializable components like the trace
        if key not in ['trace', 'model', 'idata']:
            try:
                # Test if the item is JSON serializable
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                # Skip items that can't be serialized
                logger.debug(f"Skipping non-serializable result component: {key}")
    
    # Save remaining serializable results
    if serializable_results:
        with open(output_path / "model_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Saved model results to {output_path}")


def save_comparison_results(
    comparison_data: List[Dict[str, Any]], 
    output_dir: str,
    filename: str = "comparison_results.csv"
) -> None:
    """Save comparison results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path / filename, index=False)
    
    # Generate summary statistics
    summary = {
        'mean_difference': float(comparison_df['difference'].mean()) if 'difference' in comparison_df.columns else None,
        'median_difference': float(comparison_df['difference'].median()) if 'difference' in comparison_df.columns else None,
        'min_difference': float(comparison_df['difference'].min()) if 'difference' in comparison_df.columns else None,
        'max_difference': float(comparison_df['difference'].max()) if 'difference' in comparison_df.columns else None,
        'std_difference': float(comparison_df['difference'].std()) if 'difference' in comparison_df.columns else None,
        'sample_size': len(comparison_df)
    }
    
    # Save summary
    with open(output_path / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Saved comparison results to {output_path}")


def create_comparison_data(
    model1_results: Dict[str, float], 
    model2_results: Dict[str, float],
    model1_name: str = "model1",
    model2_name: str = "model2"
) -> List[Dict[str, Any]]:
    """Create comparison data from two sets of model results."""
    comparison_data = []
    
    for item_id in set(model1_results.keys()) | set(model2_results.keys()):
        model1_value = model1_results.get(item_id, np.nan)
        model2_value = model2_results.get(item_id, np.nan)
        
        comparison_data.append({
            'id': item_id,
            model1_name: model1_value,
            model2_name: model2_value,
            'difference': model1_value - model2_value if not np.isnan(model1_value) and not np.isnan(model2_value) else np.nan
        })
    
    return comparison_data 