#!/usr/bin/env python3
"""
Command-line interface for generating synthetic retail data.

This script provides a simple command-line interface to the simulation module
for generating synthetic retail data with known elasticities for testing.
"""

import os
import argparse
import logging
from datetime import datetime

from simulation import generate_synthetic_data

def setup_logging(debug: bool = False):
    """Configure logging for the data generation process."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """
    Main entry point for the data generation script.
    Parses command-line arguments and generates synthetic data.
    """
    parser = argparse.ArgumentParser(
        description='Generate synthetic retail data for SKU elasticity analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data size parameters
    parser.add_argument('--n-observations', type=int, default=50000,
                      help='Number of sales transactions to generate')
    parser.add_argument('--n-skus', type=int, default=200,
                      help='Number of unique SKUs')
    parser.add_argument('--n-product-classes', type=int, default=10,
                      help='Number of product classes')
    parser.add_argument('--n-stores', type=int, default=20,
                      help='Number of stores')
    
    # Date range parameters
    parser.add_argument('--start-date', type=str, default="2023-01-01",
                      help='Start date for transactions (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default="2023-12-31",
                      help='End date for transactions (YYYY-MM-DD)')
    
    # Elasticity parameters
    parser.add_argument('--elasticity-mean', type=float, default=-1.2,
                      help='Mean of the true elasticity distribution')
    parser.add_argument('--elasticity-std', type=float, default=0.3,
                      help='Standard deviation of the true elasticity distribution')
    parser.add_argument('--correlation', type=float, default=-0.5,
                      help='Correlation between intercept and elasticity')
    parser.add_argument('--noise-level', type=float, default=0.2,
                      help='Standard deviation of the noise term')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, default=None,
                      help='File path to save the data (default: synthetic_YYYY-MM-DD.parquet)')
    parser.add_argument('--output-dir', type=str, default="Retail/data",
                      help='Directory to save output files')
    
    # Predefined data sizes
    parser.add_argument('--small', action='store_true',
                      help='Generate a small dataset (10K observations, 50 SKUs)')
    parser.add_argument('--medium', action='store_true',
                      help='Generate a medium dataset (50K observations, 200 SKUs)')
    parser.add_argument('--large', action='store_true',
                      help='Generate a large dataset (200K observations, 500 SKUs)')
    
    # Other options
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default output file if not specified
    if args.output_file is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        args.output_file = os.path.join(args.output_dir, f"synthetic_{current_date}.parquet")
    elif not os.path.isabs(args.output_file):
        args.output_file = os.path.join(args.output_dir, args.output_file)
    
    # Generate data based on size flag
    if args.small:
        logger.info("Generating small dataset")
        generate_synthetic_data(
            n_observations=10000,
            n_skus=50,
            n_product_classes=5,
            n_stores=10,
            true_elasticity_mean=args.elasticity_mean,
            true_elasticity_std=args.elasticity_std,
            correlation=args.correlation,
            noise_level=args.noise_level,
            output_file=args.output_file
        )
    elif args.medium:
        logger.info("Generating medium dataset")
        generate_synthetic_data(
            n_observations=50000,
            n_skus=200,
            n_product_classes=10,
            n_stores=20,
            true_elasticity_mean=args.elasticity_mean,
            true_elasticity_std=args.elasticity_std,
            correlation=args.correlation,
            noise_level=args.noise_level,
            output_file=args.output_file
        )
    elif args.large:
        logger.info("Generating large dataset")
        generate_synthetic_data(
            n_observations=200000,
            n_skus=500,
            n_product_classes=20,
            n_stores=50,
            true_elasticity_mean=args.elasticity_mean,
            true_elasticity_std=args.elasticity_std,
            correlation=args.correlation,
            noise_level=args.noise_level,
            output_file=args.output_file
        )
    else:
        # Use custom parameters
        logger.info("Generating custom dataset")
        generate_synthetic_data(
            n_observations=args.n_observations,
            n_skus=args.n_skus,
            n_product_classes=args.n_product_classes,
            n_stores=args.n_stores,
            start_date=args.start_date,
            end_date=args.end_date,
            true_elasticity_mean=args.elasticity_mean,
            true_elasticity_std=args.elasticity_std,
            correlation=args.correlation,
            noise_level=args.noise_level,
            output_file=args.output_file
        )
    
    logger.info("Data generation complete.")

if __name__ == "__main__":
    main() 