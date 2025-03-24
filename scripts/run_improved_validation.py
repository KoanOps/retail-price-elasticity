#!/usr/bin/env python3
"""
Improved Validation Script for Elasticity Models.

This script runs enhanced validation for elasticity models with:
1. Promotional features 
2. Stratified sampling to ensure sufficient observations per SKU
3. Cross-validation for stability analysis
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.simulation import generate_synthetic_data
from model.model_runner import ModelRunner
from utils.analysis.model_validation import cross_validate_elasticity_model
from data.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Retail_Elasticity')

def run_improved_validation(
    model_type: str = "bayesian",
    n_observations: int = 20000,
    n_skus: int = 100,
    min_observations_per_sku: int = 15,
    n_draws: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    use_promo: bool = True,
    cross_validate: bool = True,
    n_folds: int = 5,
    results_dir: str = "results/improved_validation"
):
    """
    Run improved validation with promotional features and sufficient observations per SKU.
    
    Args:
        model_type: Model type (bayesian or linear)
        n_observations: Total observations in synthetic dataset
        n_skus: Number of SKUs in synthetic dataset
        min_observations_per_sku: Minimum observations required per SKU
        n_draws: MCMC draws for Bayesian model
        n_tune: MCMC tuning steps for Bayesian model
        n_chains: Number of MCMC chains for Bayesian model
        target_accept: Target acceptance rate for MCMC sampling
        use_promo: Whether to include promotional features
        cross_validate: Whether to run cross-validation
        n_folds: Number of folds for cross-validation
        results_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate synthetic data with promotional features if requested
    logger.info(f"Generating synthetic data with {n_observations} observations, {n_skus} SKUs")
    
    sim_args = {
        "n_observations": n_observations,
        "n_skus": n_skus,
        "n_product_classes": 5,
        "true_elasticity_mean": -1.2,
        "true_elasticity_std": 0.3,
        "output_file": os.path.join(results_dir, "synthetic_data.parquet")
    }
    
    if use_promo:
        sim_args.update({
            "promo_frequency": 0.15,
            "promo_discount_mean": 0.3,
            "promo_elasticity_boost": 0.5
        })
        
    data = generate_synthetic_data(**sim_args)
    
    # Save synthetic data
    data_file = os.path.join(results_dir, "synthetic_data.parquet")
    data.to_parquet(data_file)
    
    logger.info(f"Saved synthetic data with {len(data)} observations to {data_file}")
    
    # Setup model parameters
    model_kwargs = {
        "n_draws": n_draws,
        "n_tune": n_tune,
        "n_chains": n_chains,
        "target_accept": target_accept
    }
    
    # Create data loader with stratified sampling and column mapping for synthetic data
    loader = DataLoader(
        data_path=data_file,
        sku_col="SKU",
        price_col="Price_Per_Unit",
        quantity_col="Qty_Sold",
        product_class_col="Product_Class",
        date_col="Transaction_Date"
    )
    
    # Run standard validation
    standard_results_dir = os.path.join(results_dir, "standard_validation")
    os.makedirs(standard_results_dir, exist_ok=True)
    
    # Sample data ensuring minimum observations per SKU
    sampled_data = loader.load_data(
        sample_frac=0.2,  # Sample 20% of overall data
        min_observations_per_sku=min_observations_per_sku,
        preprocess=True,
        add_date_features=True,
        log_transform_price=True,
        log_transform_quantity=True
    )
    
    # Save sampled data
    sampled_file = os.path.join(standard_results_dir, "sampled_data.parquet")
    sampled_data.to_parquet(sampled_file)
    
    logger.info(f"Saved sampled data with {len(sampled_data)} observations to {sampled_file}")
    
    # Run model on sampled data
    runner = ModelRunner(results_dir=standard_results_dir)
    
    logger.info(f"Running {model_type} model on sampled data")
    
    # Set column mappings for ModelRunner
    data_config = {
        "price_col": "Price_Per_Unit",
        "quantity_col": "Qty_Sold",
        "sku_col": "SKU",
        "product_class_col": "Product_Class",
        "date_col": "Transaction_Date"
    }
    
    # Update model kwargs with data config
    model_kwargs["data_config"] = data_config
    
    standard_results = runner.run_analysis(
        data_path=sampled_file,
        model_type=model_type,
        **model_kwargs
    )
    
    if cross_validate:
        logger.info("Running cross-validation for stability analysis")
        
        cv_results_dir = os.path.join(results_dir, "cross_validation")
        os.makedirs(cv_results_dir, exist_ok=True)
        
        # Run cross-validation
        cv_results = cross_validate_elasticity_model(
            data=data,
            model_type=model_type,
            date_col="Transaction_Date",
            n_folds=n_folds,
            min_observations_per_sku=min_observations_per_sku,
            model_kwargs=model_kwargs,
            results_dir=cv_results_dir
        )
        
        # Summarize cross-validation results
        if "overall_stability" in cv_results:
            stability = cv_results["overall_stability"]
            logger.info(f"Cross-validation results:")
            logger.info(f"  SKUs analyzed: {stability['num_skus_analyzed']}")
            logger.info(f"  Mean CV: {stability['mean_cv']:.4f}")
            logger.info(f"  Stable SKUs: {stability['num_stable_skus']} ({stability['stability_rate']:.1%})")
        else:
            logger.warning("Cross-validation did not produce stability metrics")
    
    # Return results
    return {
        "standard_results": standard_results,
        "cv_results": cv_results if cross_validate else None
    }

def main():
    parser = argparse.ArgumentParser(description="Run improved elasticity model validation")
    
    parser.add_argument(
        "--model-type", 
        choices=["bayesian", "linear"],
        default="bayesian",
        help="Type of model to validate"
    )
    
    parser.add_argument(
        "--observations", 
        type=int, 
        default=20000,
        help="Number of observations in synthetic dataset"
    )
    
    parser.add_argument(
        "--skus", 
        type=int, 
        default=100,
        help="Number of SKUs in synthetic dataset"
    )
    
    parser.add_argument(
        "--min-obs-per-sku", 
        type=int, 
        default=15,
        help="Minimum observations required per SKU"
    )
    
    parser.add_argument(
        "--draws", 
        type=int, 
        default=1000,
        help="Number of MCMC draws for Bayesian model"
    )
    
    parser.add_argument(
        "--tune", 
        type=int, 
        default=1000,
        help="Number of MCMC tuning steps for Bayesian model"
    )
    
    parser.add_argument(
        "--chains", 
        type=int, 
        default=4,
        help="Number of MCMC chains for Bayesian model"
    )
    
    parser.add_argument(
        "--target-accept", 
        type=float, 
        default=0.95,
        help="Target acceptance rate for MCMC sampling"
    )
    
    parser.add_argument(
        "--no-promo", 
        action="store_true",
        help="Don't include promotional features in simulation"
    )
    
    parser.add_argument(
        "--no-cv", 
        action="store_true",
        help="Skip cross-validation stability analysis"
    )
    
    parser.add_argument(
        "--folds", 
        type=int, 
        default=5,
        help="Number of folds for cross-validation"
    )
    
    parser.add_argument(
        "--results-dir", 
        default="results/improved_validation",
        help="Directory to save validation results"
    )
    
    args = parser.parse_args()
    
    # Run improved validation
    run_improved_validation(
        model_type=args.model_type,
        n_observations=args.observations,
        n_skus=args.skus,
        min_observations_per_sku=args.min_obs_per_sku,
        n_draws=args.draws,
        n_tune=args.tune,
        n_chains=args.chains,
        target_accept=args.target_accept,
        use_promo=not args.no_promo,
        cross_validate=not args.no_cv,
        n_folds=args.folds,
        results_dir=args.results_dir
    )
    
    logger.info(f"Improved validation completed. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main() 