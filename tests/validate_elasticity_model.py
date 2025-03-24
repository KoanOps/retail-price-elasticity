#!/usr/bin/env python
"""
Elasticity Model Validation Tool for Synthetic Testing.

This script provides a complete command-line tool for validating price elasticity models 
against synthetic data with known ground-truth elasticities. It supports configurable 
test scenarios and detailed performance metrics.

PURPOSE:
- Validate elasticity models against data with known true elasticities
- Generate performance metrics and visualization of model accuracy
- Support hyperparameter tuning and model comparison
- Provide a standardized benchmark for model improvements

USAGE:
    python validate_elasticity_model.py 
        --model bayesian
        --observations 10000 
        --skus 50 
        --sample-frac 0.3
        --draws 1000 
        --tune 500
        --results-dir results/validation_test

VALIDATION METRICS:
- Mean Absolute Error (MAE): Average absolute difference between true and estimated elasticities
- Root Mean Squared Error (RMSE): Square root of the average squared differences
- Correlation: Correlation coefficient between true and estimated values
- Sign Match Rate: Proportion of elasticities with correctly predicted sign (+ or -)
- Relative Error: Average relative difference between true and estimated values

ASSUMPTIONS:
- The BayesianModel implementation is available and properly configured
- System has sufficient memory and processing power for the specified parameters
- Results directory is writable for saving outputs
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.analysis.model_validation import validate_with_simulated_data, perform_holdout_validation
from data.simulation import generate_synthetic_data
from utils.logging_utils import logger
from model.constants import DEFAULT_CHAINS

def run_simulation_validation(
    model_type: str = "bayesian",
    n_observations: int = 5000,
    n_skus: int = 50,
    n_product_classes: int = 5,
    true_elasticity_mean: float = -1.2,
    true_elasticity_std: float = 0.3,
    results_dir: str = "validation/simulated",
    sample_frac: Optional[float] = 0.2,
    n_draws: int = 500,
    n_tune: int = 250
) -> Dict[str, Any]:
    """
    Run validation against simulated data with known elasticities.
    
    Args:
        model_type: Type of model to validate
        n_observations: Number of observations in synthetic dataset
        n_skus: Number of SKUs
        n_product_classes: Number of product classes
        true_elasticity_mean: Mean of true elasticity distribution
        true_elasticity_std: Standard deviation of true elasticity distribution
        results_dir: Directory to save validation results
        sample_frac: Optional fraction of data to sample (for faster testing)
        n_draws: Number of MCMC draws for Bayesian model
        n_tune: Number of tuning steps for Bayesian model
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Running validation for {model_type} model against simulated data")
    
    # Configure model parameters
    model_kwargs = {
        "sample_frac": sample_frac,
        "n_draws": n_draws,
        "n_tune": n_tune,
        "n_chains": DEFAULT_CHAINS  # Use value from constants
    }
    
    # Run validation
    results = validate_with_simulated_data(
        model_type=model_type,
        n_observations=n_observations,
        n_skus=n_skus,
        n_product_classes=n_product_classes,
        true_elasticity_mean=true_elasticity_mean,
        true_elasticity_std=true_elasticity_std,
        results_dir=results_dir,
        model_kwargs=model_kwargs,
        save_plots=True
    )
    
    # Print results
    if "validation_metrics" in results:
        logger.info("Validation Metrics:")
        for metric, value in results["validation_metrics"].items():
            logger.info(f"  {metric}: {value}")
    else:
        logger.error(f"Validation failed: {results.get('error', 'Unknown error')}")
    
    return results

def main():
    """Parse arguments and run validation."""
    parser = argparse.ArgumentParser(description="Validate elasticity model against simulated data")
    
    parser.add_argument(
        "--model-type", 
        default="bayesian", 
        choices=["bayesian", "linear"],
        help="Type of model to validate"
    )
    
    parser.add_argument(
        "--observations", 
        type=int, 
        default=5000,
        help="Number of observations in synthetic dataset"
    )
    
    parser.add_argument(
        "--skus", 
        type=int, 
        default=50,
        help="Number of SKUs in synthetic dataset"
    )
    
    parser.add_argument(
        "--elasticity-mean", 
        type=float, 
        default=-1.2,
        help="Mean of true elasticity distribution"
    )
    
    parser.add_argument(
        "--elasticity-std", 
        type=float, 
        default=0.3,
        help="Standard deviation of true elasticity distribution"
    )
    
    parser.add_argument(
        "--results-dir", 
        default="validation/simulated",
        help="Directory to save validation results"
    )
    
    parser.add_argument(
        "--sample-frac", 
        type=float, 
        default=0.2,
        help="Fraction of data to sample for validation"
    )
    
    parser.add_argument(
        "--draws", 
        type=int, 
        default=500,
        help="Number of MCMC draws for Bayesian model"
    )
    
    parser.add_argument(
        "--tune", 
        type=int, 
        default=250,
        help="Number of tuning steps for Bayesian model"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    results = run_simulation_validation(
        model_type=args.model_type,
        n_observations=args.observations,
        n_skus=args.skus,
        true_elasticity_mean=args.elasticity_mean,
        true_elasticity_std=args.elasticity_std,
        results_dir=args.results_dir,
        sample_frac=args.sample_frac,
        n_draws=args.draws,
        n_tune=args.tune
    )
    
    # Save results to file
    results_file = os.path.join(args.results_dir, "validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved validation results to {results_file}")
    
    # Return success/failure
    return 0 if "validation_metrics" in results else 1

if __name__ == "__main__":
    sys.exit(main()) 