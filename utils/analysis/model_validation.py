"""
Model Validation Framework for Retail Elasticity Models.

This module provides comprehensive tools for validating price elasticity models
through two main approaches:
1. Synthetic Data Validation - Using simulated data with known elasticities
2. Holdout Validation - Using real data split into training and testing sets

PURPOSE:
This validation framework allows researchers and data scientists to:
- Verify a model's ability to recover known elasticities from simulated data
- Compare performance across different models or parameter settings
- Generate diagnostic plots and metrics for model evaluation
- Quantify uncertainty in elasticity estimation

ASSUMPTIONS:
- For synthetic validation, the true elasticities are known and fixed
- Elasticities are generally negative (price increases decrease demand)
- SKU identifiers remain consistent between true and estimated results
- Data follows the expected structure with price, quantity, SKU, and product class columns

EDGE CASES:
- Missing columns in validation data will raise explicit errors
- Empty validation results are handled gracefully with appropriate error messages
- SKU mismatches between true and estimated values are reported in metrics
- Zero or positive true elasticity values may cause issues with relative error calculation

USAGE:
For standard validation run:
    mkdir -p results/validation_test && python tests/test_model_validation.py \
    --observations 10000 --skus 50 --sample-frac 0.3 --draws 1000 --tune 500 \
    --results-dir results/validation_test
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import logging
import traceback

from model.model_runner import ModelRunner
from data.simulation import generate_synthetic_data
from utils.logging_utils import logger, log_step
from config.config_manager import ConfigManager

def validate_with_simulated_data(
    model_type: str = "bayesian",
    n_observations: int = 10000,
    n_skus: int = 100,
    n_product_classes: int = 8,
    true_elasticity_mean: float = -1.2,
    true_elasticity_std: float = 0.3,
    results_dir: str = "validation_results",
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Validate model performance using simulated data with known elasticities.
    
    Args:
        model_type: Type of model to validate (bayesian, linear, etc.)
        n_observations: Number of observations in synthetic dataset
        n_skus: Number of SKUs in synthetic dataset
        n_product_classes: Number of product classes
        true_elasticity_mean: Mean of true elasticity distribution
        true_elasticity_std: Standard deviation of true elasticity distribution
        results_dir: Directory to save validation results
        model_kwargs: Additional parameters for the model
        save_plots: Whether to save validation plots
        
    Returns:
        Dictionary with validation results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate synthetic data with known elasticities
    logger.info(f"Generating synthetic data with {n_observations} observations, {n_skus} SKUs")
    
    data_file = os.path.join(results_dir, "synthetic_validation_data.parquet")
    metadata_file = os.path.join(results_dir, "synthetic_validation_metadata.json")
    
    # Generate data if files don't exist
    if not os.path.exists(data_file) or not os.path.exists(metadata_file):
        synthetic_data = generate_synthetic_data(
            n_observations=n_observations,
            n_skus=n_skus,
            n_product_classes=n_product_classes,
            true_elasticity_mean=true_elasticity_mean,
            true_elasticity_std=true_elasticity_std,
            output_file=data_file
        )
        
        # Metadata should have been saved by generate_synthetic_data
        if not os.path.exists(metadata_file):
            metadata_file = os.path.splitext(data_file)[0] + "_metadata.json"
    
    # Load data and metadata
    data = pd.read_parquet(data_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Define both standardized names and the original names the DataLoader expects
    original_to_standardized = {
        'Transaction_Date': 'Date',
        'SKU': 'sku',
        'Product_Class': 'product_class',
        'Qty_Sold': 'quantity',
        'Price_Per_Unit': 'price',
        'Store_ID': 'store_id'
    }
    
    # Also define mappings to the names expected by DataLoader
    original_to_dataloader = {
        'SKU': 'SKU_Coded',
        'Product_Class': 'Product_Class_Code',
        'Qty_Sold': 'Qty_Sold',  # Keep original name
        'Price_Per_Unit': 'Price_Per_Unit',  # Keep original name
    }
    
    # Create a new DataFrame with both standardized and original column names
    transformed_data = data.copy()
    
    # First add standardized columns
    for orig_col, std_col in original_to_standardized.items():
        if orig_col in transformed_data.columns:
            transformed_data[std_col] = transformed_data[orig_col]
    
    # Then add DataLoader expected columns
    for orig_col, dl_col in original_to_dataloader.items():
        if orig_col in transformed_data.columns:
            transformed_data[dl_col] = transformed_data[orig_col]
    
    # Save the transformed data
    transformed_data_file = os.path.join(results_dir, "transformed_validation_data.parquet")
    transformed_data.to_parquet(transformed_data_file)
    
    logger.info(f"Transformed data columns: {list(transformed_data.columns)}")
    
    # Extract true elasticities
    true_elasticities = metadata.get("true_elasticities", {})
    if not true_elasticities:
        logger.warning("Metadata does not contain true elasticities")
        return {"error": "No true elasticities found in metadata"}
    
    # For elasticity validation, we need to map the SKU_Coded values back to original SKU values
    # Create a mapping from the new column names to the original ones for the elasticities
    sku_mapping = {}
    if 'SKU' in data.columns and 'SKU_Coded' in transformed_data.columns:
        for orig_sku, coded_sku in zip(data['SKU'], transformed_data['SKU_Coded']):
            sku_mapping[coded_sku] = orig_sku
    
    # Initialize model runner with appropriate parameters
    model_kwargs = model_kwargs or {}
    model_results_dir = os.path.join(results_dir, "model_results")
    
    # Create a configuration manager with appropriate settings
    config_manager = ConfigManager()
    
    # Set up app config
    config_manager.app_config.results_dir = model_results_dir
    
    # Set up model config with sampling parameters
    config_manager.model_config.n_draws = model_kwargs.get('n_draws', 1000)
    config_manager.model_config.n_tune = model_kwargs.get('n_tune', 500)
    config_manager.model_config.n_chains = model_kwargs.get('n_chains', 2)
    
    # Create runner with configuration
    runner = ModelRunner(results_dir=model_results_dir, config_manager=config_manager)
    
    # Run model on synthetic data
    logger.info(f"Running {model_type} model on synthetic data")
    try:
        # Sample data if requested
        sample_frac = model_kwargs.get('sample_frac', None)
        data_file_to_use = transformed_data_file
        
        if sample_frac is not None and sample_frac < 1.0:
            # Sample data for faster processing
            sampled_data = transformed_data.sample(frac=sample_frac, random_state=42)
            sampled_data_file = os.path.join(results_dir, "sampled_validation_data.parquet")
            sampled_data.to_parquet(sampled_data_file)
            data_file_to_use = sampled_data_file
            logger.info(f"Created sampled data with {len(sampled_data)} rows")
        
        # Run analysis using run_analysis method
        results = runner.run_analysis(
            data_path=data_file_to_use,
            model_type=model_type,
            **model_kwargs
        )
        
        # Extract estimated elasticities
        estimated_elasticities = results.get("elasticities", {})
        if not estimated_elasticities:
            logger.warning("Model did not produce elasticity estimates")
            return {"error": "No elasticities in model results"}
        
        # Map the estimated elasticities back to original SKUs if needed
        if sku_mapping and estimated_elasticities:
            mapped_elasticities = {}
            for coded_sku, elasticity in estimated_elasticities.items():
                orig_sku = sku_mapping.get(coded_sku, coded_sku)
                mapped_elasticities[orig_sku] = elasticity
            estimated_elasticities = mapped_elasticities
        
        # Calculate validation metrics
        validation_metrics = calculate_validation_metrics(
            true_elasticities=true_elasticities,
            estimated_elasticities=estimated_elasticities
        )
        
        # Create validation report
        validation_report = {
            "validation_metrics": validation_metrics,
            "model_type": model_type,
            "data_params": {
                "n_observations": n_observations,
                "n_skus": n_skus,
                "n_product_classes": n_product_classes,
                "true_elasticity_mean": true_elasticity_mean,
                "true_elasticity_std": true_elasticity_std
            },
            "model_params": model_kwargs
        }
        
        # Save validation report
        report_file = os.path.join(results_dir, "validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Saved validation report to {report_file}")
        
        # Generate validation plots
        if save_plots:
            create_validation_plots(
                true_elasticities=true_elasticities,
                estimated_elasticities=estimated_elasticities,
                output_dir=results_dir
            )
        
        return validation_report
        
    except Exception as e:
        logger.error(f"Error during model validation: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def calculate_validation_metrics(
    true_elasticities: Dict[str, float],
    estimated_elasticities: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate metrics to evaluate model performance.
    
    Args:
        true_elasticities: Dictionary of SKU to true elasticity
        estimated_elasticities: Dictionary of SKU to estimated elasticity
        
    Returns:
        Dictionary with validation metrics
    """
    # Get common SKUs
    common_skus = set(true_elasticities.keys()) & set(estimated_elasticities.keys())
    n_common = len(common_skus)
    
    if n_common == 0:
        logger.warning("No common SKUs between true and estimated elasticities")
        return {
            "error": "No common SKUs to compare",
            "match_rate": 0.0
        }
    
    # Create arrays of true and estimated values
    true_values = [true_elasticities[sku] for sku in common_skus]
    estimated_values = [estimated_elasticities[sku] for sku in common_skus]
    
    # Convert to numpy arrays
    true_array = np.array(true_values)
    estimated_array = np.array(estimated_values)
    
    # Calculate metrics
    mae = np.mean(np.abs(true_array - estimated_array))
    mse = np.mean((true_array - estimated_array) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate correlation
    correlation = np.corrcoef(true_array, estimated_array)[0, 1]
    
    # Calculate sign match rate (both negative or both positive)
    sign_match = np.mean((true_array < 0) == (estimated_array < 0))
    
    # Calculate relative error
    relative_error = np.mean(np.abs((true_array - estimated_array) / true_array))
    
    # Create metrics dictionary
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "sign_match_rate": float(sign_match),
        "relative_error": float(relative_error),
        "n_skus_compared": n_common,
        "match_rate": n_common / max(len(true_elasticities), len(estimated_elasticities))
    }
    
    return metrics

def create_validation_plots(
    true_elasticities: Dict[str, float],
    estimated_elasticities: Dict[str, float],
    output_dir: str = "validation_results"
) -> None:
    """
    Create plots to visualize validation results.
    
    Args:
        true_elasticities: Dictionary of SKU to true elasticity
        estimated_elasticities: Dictionary of SKU to estimated elasticity
        output_dir: Directory to save the plots
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get common SKUs
        common_skus = set(true_elasticities.keys()) & set(estimated_elasticities.keys())
        
        if not common_skus:
            logger.warning("No common SKUs for validation plots")
            return
        
        # Create arrays of true and estimated values
        true_values = [true_elasticities[sku] for sku in common_skus]
        estimated_values = [estimated_elasticities[sku] for sku in common_skus]
        
        # Plot 1: Scatter plot of true vs estimated elasticities
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        # Plot the scatter
        sns.scatterplot(x=true_values, y=estimated_values, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(min(true_values), min(estimated_values))
        max_val = max(max(true_values), max(estimated_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        plt.xlabel("True Elasticity")
        plt.ylabel("Estimated Elasticity")
        plt.title("True vs. Estimated Price Elasticities")
        
        # Add correlation annotation
        corr = np.corrcoef(true_values, estimated_values)[0, 1]
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "elasticity_scatter.png"))
        plt.close()
        
        # Plot 2: Distribution of estimation errors
        plt.figure(figsize=(10, 6))
        
        # Calculate errors
        errors = np.array(estimated_values) - np.array(true_values)
        
        # Plot histogram of errors
        sns.histplot(errors, kde=True)
        
        # Add labels and title
        plt.xlabel("Estimation Error (Estimated - True)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Elasticity Estimation Errors")
        
        # Add stats annotations
        plt.annotate(f"Mean Error: {np.mean(errors):.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        plt.annotate(f"Std Dev: {np.std(errors):.3f}", xy=(0.05, 0.90), xycoords='axes fraction')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_distribution.png"))
        plt.close()
        
        # Plot 3: True and estimated distributions
        plt.figure(figsize=(12, 6))
        
        # Plot KDE of both distributions
        sns.kdeplot(true_values, label="True Elasticities", color="blue")
        sns.kdeplot(estimated_values, label="Estimated Elasticities", color="red")
        
        # Add labels and title
        plt.xlabel("Elasticity")
        plt.ylabel("Density")
        plt.title("Distribution of True vs. Estimated Elasticities")
        plt.legend()
        
        # Add stats annotations
        plt.annotate(f"True Mean: {np.mean(true_values):.3f}", xy=(0.05, 0.95), xycoords='axes fraction', color="blue")
        plt.annotate(f"Est. Mean: {np.mean(estimated_values):.3f}", xy=(0.05, 0.90), xycoords='axes fraction', color="red")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "elasticity_distributions.png"))
        plt.close()
        
        logger.info(f"Created validation plots in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating validation plots: {str(e)}")

def perform_holdout_validation(
    data: pd.DataFrame,
    model_type: str = "bayesian",
    test_size: float = 0.3,
    time_based_split: bool = True,
    date_column: str = "Transaction_Date",
    results_dir: str = "holdout_results",
    model_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform holdout validation by splitting data into train and test sets.
    
    Args:
        data: Full dataset
        model_type: Type of model to validate
        test_size: Proportion of data to use for testing
        time_based_split: Whether to split data by time
        date_column: Column name for date (if time_based_split is True)
        results_dir: Directory to save validation results
        model_kwargs: Additional parameters for the model
        
    Returns:
        Dictionary with validation results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Clone the dataframe
    data = data.copy()
    
    # Split data into train and test sets
    if time_based_split and date_column in data.columns:
        logger.info("Performing time-based train/test split")
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        data = data.sort_values(by=date_column)
        
        # Find split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split data
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
    else:
        logger.info("Performing random train/test split")
        
        # Shuffle data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
    
    logger.info(f"Train set: {len(train_data)} rows, Test set: {len(test_data)} rows")
    
    # Save split data
    train_file = os.path.join(results_dir, "train_data.parquet")
    test_file = os.path.join(results_dir, "test_data.parquet")
    
    train_data.to_parquet(train_file)
    test_data.to_parquet(test_file)
    
    logger.info(f"Saved train data to {train_file} and test data to {test_file}")
    
    # Train model on training data
    model_kwargs = model_kwargs or {}
    train_results_dir = os.path.join(results_dir, "train_results")
    
    runner = ModelRunner(
        model_type=model_type,
        results_dir=train_results_dir,
        **model_kwargs
    )
    
    logger.info(f"Training {model_type} model on training data")
    train_results = runner.run(data=train_data)
    
    # Get elasticity estimates from training
    elasticity_estimates = train_results.get("elasticities", {})
    
    if not elasticity_estimates:
        logger.warning("Model did not produce elasticity estimates")
        return {"error": "No elasticities in model results"}
    
    # TODO: Implement evaluation on test data
    # This would depend on the specific validation metric for the holdout set
    # For example, predictive accuracy, or comparison to elasticities estimated on test data
    
    # Placeholder for holdout metrics
    holdout_metrics = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "n_elasticities": len(elasticity_estimates)
    }
    
    # Create validation report
    validation_report = {
        "holdout_metrics": holdout_metrics,
        "model_type": model_type,
        "time_based_split": time_based_split,
        "test_size": test_size,
        "model_params": model_kwargs
    }
    
    # Save validation report
    report_file = os.path.join(results_dir, "holdout_report.json")
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Saved holdout validation report to {report_file}")
    
    return validation_report

def cross_validate_elasticity_model(
    data: pd.DataFrame,
    model_type: str = "bayesian",
    date_col: str = "Transaction_Date", 
    n_folds: int = 5,
    min_observations_per_sku: int = 10,
    model_kwargs: Optional[Dict[str, Any]] = None,
    results_dir: str = "results/cross_validation"
) -> Dict[str, Any]:
    """
    Perform time-based cross-validation of elasticity models.
    
    This function splits the data into time-based folds and evaluates model
    stability by comparing elasticity estimates across different time periods.
    
    Args:
        data: Transaction data for model validation
        model_type: Type of model to validate (bayesian, linear, etc.)
        date_col: Name of the column containing dates
        n_folds: Number of time-based folds for cross-validation
        min_observations_per_sku: Minimum observations required per SKU in each fold
        model_kwargs: Additional parameters for the model
        results_dir: Directory to save cross-validation results
        
    Returns:
        Dictionary with cross-validation results
    """
    # Ensure date column is datetime
    if data[date_col].dtype != 'datetime64[ns]':
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Sort data by date
    data = data.sort_values(by=date_col)
    
    # Get date range
    min_date = data[date_col].min()
    max_date = data[date_col].max()
    date_range = (max_date - min_date).days
    
    # Create time-based folds
    fold_size = date_range // n_folds
    fold_boundaries = [min_date + pd.Timedelta(days=fold_size * i) for i in range(n_folds + 1)]
    
    logger.info(f"Cross-validation with {n_folds} folds")
    logger.info(f"Date range: {min_date.date()} to {max_date.date()} ({date_range} days)")
    
    # Initialize model results storage
    fold_results = []
    sku_elasticities = {}
    
    # Default model params if none provided
    model_kwargs = model_kwargs or {}
    if 'sample_frac' not in model_kwargs:
        model_kwargs['sample_frac'] = 1.0  # Use all data in each fold
    
    # Run model for each fold
    for i in range(n_folds):
        fold_start = fold_boundaries[i]
        fold_end = fold_boundaries[i + 1]
        
        logger.info(f"Fold {i+1}: {fold_start.date()} to {fold_end.date()}")
        
        # Filter data for this fold
        fold_data = data[(data[date_col] >= fold_start) & (data[date_col] < fold_end)]
        
        # Skip fold if insufficient data
        if len(fold_data) < 100:
            logger.warning(f"Insufficient data in fold {i+1}, skipping")
            continue
        
        # Create fold directory
        fold_dir = os.path.join(results_dir, f"fold_{i+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Check SKU observation counts and keep only those with sufficient data
        sku_counts = fold_data['sku'].value_counts()
        skus_to_keep = sku_counts[sku_counts >= min_observations_per_sku].index
        if len(skus_to_keep) < 10:
            logger.warning(f"Fold {i+1} has fewer than 10 SKUs with sufficient data, skipping")
            continue
            
        fold_data_filtered = fold_data[fold_data['sku'].isin(skus_to_keep)]
        logger.info(f"Fold {i+1} has {len(fold_data_filtered)} observations across {len(skus_to_keep)} SKUs")
        
        # Save fold data
        fold_data_file = os.path.join(fold_dir, "fold_data.parquet")
        fold_data_filtered.to_parquet(fold_data_file)
        
        # Run model on this fold
        try:
            # Initialize model runner with appropriate parameters
            runner = ModelRunner(results_dir=fold_dir)
            
            # Run analysis
            logger.info(f"Running {model_type} model on fold {i+1}")
            fold_model_results = runner.run_analysis(
                data_path=fold_data_file,
                model_type=model_type,
                **model_kwargs
            )
            
            # Extract and store elasticity estimates
            if "elasticities" in fold_model_results:
                elasticities = fold_model_results["elasticities"]
                
                # Store fold results
                fold_results.append({
                    "fold": i + 1,
                    "num_observations": len(fold_data_filtered),
                    "num_skus": len(elasticities),
                    "mean_elasticity": np.mean(list(elasticities.values())),
                    "elasticities": elasticities
                })
                
                # Add to SKU tracking
                for sku, elasticity in elasticities.items():
                    if sku not in sku_elasticities:
                        sku_elasticities[sku] = []
                    sku_elasticities[sku].append(elasticity)
            else:
                logger.warning(f"No elasticity estimates found for fold {i+1}")
                
        except Exception as e:
            logger.error(f"Error processing fold {i+1}: {str(e)}")
    
    # Calculate stability metrics
    logger.info("Calculating elasticity stability metrics")
    
    # Only keep SKUs that appear in at least half the folds
    min_fold_presence = max(2, n_folds // 2)
    stable_skus = {sku: values for sku, values in sku_elasticities.items() 
                  if len(values) >= min_fold_presence}
    
    if not stable_skus:
        logger.warning("No SKUs present in enough folds for stability analysis")
        return {
            "error": "Insufficient data for cross-validation",
            "fold_results": fold_results
        }
    
    # Calculate per-SKU stability metrics
    sku_stability = {}
    for sku, elasticities in stable_skus.items():
        # Ensure elasticities are numeric (could be dicts for models with promo features)
        if isinstance(elasticities[0], dict) and 'weighted' in elasticities[0]:
            # Extract weighted values
            elasticity_values = [e['weighted'] for e in elasticities]
        else:
            elasticity_values = elasticities
            
        sku_stability[sku] = {
            "mean": float(np.mean(elasticity_values)),
            "std": float(np.std(elasticity_values)),
            "cv": float(np.std(elasticity_values) / abs(np.mean(elasticity_values))),
            "min": float(np.min(elasticity_values)),
            "max": float(np.max(elasticity_values)),
            "range": float(np.max(elasticity_values) - np.min(elasticity_values)),
            "num_folds": len(elasticity_values)
        }
    
    # Calculate overall stability metrics
    cv_values = [stats["cv"] for stats in sku_stability.values()]
    overall_stability = {
        "num_skus_analyzed": len(stable_skus),
        "mean_cv": float(np.mean(cv_values)),
        "median_cv": float(np.median(cv_values)),
        "max_cv": float(np.max(cv_values)),
        "num_stable_skus": len([cv for cv in cv_values if cv < 0.5]),  # CV < 50% considered stable
        "stability_rate": float(len([cv for cv in cv_values if cv < 0.5]) / len(cv_values))
    }
    
    # Save results
    results = {
        "fold_results": fold_results,
        "sku_stability": sku_stability,
        "overall_stability": overall_stability
    }
    
    with open(os.path.join(results_dir, "cross_validation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create stability visualization
    try:
        import matplotlib.pyplot as plt
        
        # Plot distribution of coefficient of variation
        plt.figure(figsize=(12, 6))
        plt.hist(cv_values, bins=20)
        plt.axvline(0.5, color='red', linestyle='--', label='Stability threshold (CV=0.5)')
        plt.xlabel('Coefficient of Variation')
        plt.ylabel('Count of SKUs')
        plt.title('Elasticity Stability Distribution (Lower CV = More Stable)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, "stability_distribution.png"))
        
        # Plot top 20 most unstable SKUs
        unstable_skus = sorted(
            [(sku, stats["cv"]) for sku, stats in sku_stability.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        if unstable_skus:
            plt.figure(figsize=(12, 8))
            x = [sku for sku, _ in unstable_skus]
            y = [cv for _, cv in unstable_skus]
            plt.barh(x, y)
            plt.xlabel('Coefficient of Variation')
            plt.ylabel('SKU')
            plt.title('Top 20 Most Unstable SKUs')
            plt.savefig(os.path.join(results_dir, "unstable_skus.png"))
        
    except Exception as e:
        logger.warning(f"Could not create stability visualizations: {str(e)}")
    
    # Log summary
    logger.info(f"Cross-validation complete. Analyzed {len(stable_skus)} SKUs across {n_folds} time periods")
    logger.info(f"Mean coefficient of variation: {overall_stability['mean_cv']:.4f}")
    logger.info(f"Stability rate (CV < 0.5): {overall_stability['stability_rate']:.2%}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    validation_results = validate_with_simulated_data(
        model_type="bayesian",
        n_observations=10000,
        n_skus=50,
        results_dir="validation/simulated"
    )
    
    print(f"Validation Results:")
    for metric, value in validation_results.get("validation_metrics", {}).items():
        print(f"  {metric}: {value}") 