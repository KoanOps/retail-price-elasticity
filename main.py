#!/usr/bin/env python3
"""
Main entry point for the Retail Price Elasticity Analysis.

This script provides a unified interface to various analysis functions:
1. Run a full elasticity analysis (with or without seasonality)
2. Generate detailed diagnostics for model evaluation
3. Compare models with and without seasonality
4. Summarize model features and differences

Usage:
    retail-analysis --run full  # Run full analysis
    retail-analysis --run diagnostics  # Run model diagnostics
    retail-analysis --run compare  # Compare models with/without seasonality
    retail-analysis --summarize  # Summarize model features
"""
import sys
import json
from pathlib import Path
import argparse
import traceback
import os
from dataclasses import asdict

# Import specialized functions from analysis.py
from analysis import (
    run_full_analysis,
    run_diagnostics,
    compare_models,
    summarize_model_features,
    load_analysis_data
)

from model.model_runner import ModelRunner
from utils.logging_utils import get_logger, LoggingManager
from model.exceptions import ModelError, ConfigurationError
from config.config_manager import ConfigManager
from utils.analysis.visualizers import visualize_elasticities

# Get logger for this module
logger = get_logger()

def main():
    """Main entry point for the Retail Elasticity Analysis."""
    # Parse command line arguments first
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Create configuration manager
    config_manager = setup_config(args)
    
    # Save configuration for reference
    results_config_path = Path(config_manager.app_config.results_dir) / "config.json"
    config_manager.save_config(results_config_path)
    
    # Report data path being used
    data_path = config_manager.data_config.data_path
    logger.info(f"Using data path: {data_path}")
    
    # Report column mappings if present
    if hasattr(config_manager.data_config, 'column_mappings') and config_manager.data_config.column_mappings:
        logger.info(f"Column mappings: {config_manager.data_config.column_mappings}")
    
    # Set up model runner
    runner = ModelRunner(
        results_dir=config_manager.app_config.results_dir,
        config_manager=config_manager
    )
    
    # Execute the requested operation
    try:
        if args.run == "full":
            return run_full_analysis_handler(args, config_manager, runner, data_path)
        elif args.run == "visualize":
            return run_visualize_handler(args, config_manager)
        else:
            logger.error(f"Unknown run type: {args.run}")
            return 1
    except Exception as e:
        logger.error(f"Error running {args.run} analysis: {str(e)}")
        logger.error("Analysis failed")
        return 1

def run_full_analysis_handler(args, config_manager, runner, data_path):
    """Handler for running a full analysis."""
    # Extract parameters for the analysis
    model_params = {
        'model_type': config_manager.model_config.model_type,
        'sample_frac': config_manager.data_config.sample_frac,
        'use_seasonality': config_manager.model_config.use_seasonality,
        'n_draws': config_manager.model_config.n_draws,
        'n_tune': config_manager.model_config.n_tune,
        'n_chains': config_manager.model_config.n_chains,
        'target_accept': config_manager.model_config.target_accept,
        'test_mode': args.test_mode
    }
    
    # Convert configuration objects to dictionaries for the model
    data_config = asdict(config_manager.data_config) if hasattr(config_manager.data_config, '__dataclass_fields__') else config_manager.data_config
    model_config = asdict(config_manager.model_config) if hasattr(config_manager.model_config, '__dataclass_fields__') else config_manager.model_config
    
    model_params['data_config'] = data_config
    model_params['model_config'] = model_config
    
    # Run the analysis through the model runner
    logger.info("Starting full elasticity analysis...")
    runner.run_analysis(data_path, **model_params)
    
    logger.info("Full analysis completed successfully")
    return 0

def run_visualize_handler(args, config_manager):
    """Handler for visualizing existing results."""
    logger.info("Visualizing existing results...")
    
    # Load elasticity results from the results directory
    results_path = Path(config_manager.app_config.results_dir) / "elasticities.json"
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Check if elasticities are nested under an 'elasticities' key
        if isinstance(data, dict) and "elasticities" in data:
            elasticities = data["elasticities"]
        else:
            elasticities = data  # Use the entire object if not nested
        
        # Create visualizations
        visualize_elasticities(elasticities, config_manager.app_config.results_dir)
        logger.info(f"Visualizations created in {config_manager.app_config.results_dir}")
        return 0
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_path}")
        return 1

def setup_config(args):
    """
    Set up and validate configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        ConfigManager instance
    """
    from config.config_manager import ConfigManager
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Override with command line arguments
    if args.data_path:
        config_manager.data_config.data_path = args.data_path
        
    if args.results_dir:
        config_manager.app_config.results_dir = args.results_dir
        
    config_manager.model_config.use_seasonality = args.use_seasonality
    config_manager.model_config.model_type = args.model_type
    config_manager.data_config.sample_frac = args.sample_frac
    
    # Set up test mode with test data if requested
    if args.test_mode:
        logger.info("Running in test mode with test data")
        
        # Configure test-specific settings
        args.run = "full"  # Force run type to full
        config_manager.data_config.data_path = "data/test_fixed.parquet"
        config_manager.app_config.results_dir = "results/test_run"
        config_manager.model_config.use_seasonality = False  # Simplify model for testing
        config_manager.model_config.n_draws = 100  # Reduce MCMC parameters for faster test run
        config_manager.model_config.n_tune = 50
        config_manager.model_config.n_chains = 1
        
        logger.info(f"Test mode configuration: data={config_manager.data_config.data_path}, results={config_manager.app_config.results_dir}")
    
    # Add MCMC sampling parameters to config if provided
    if hasattr(args, 'draws'):
        config_manager.model_config.n_draws = args.draws
    if hasattr(args, 'tune'):
        config_manager.model_config.n_tune = args.tune
    if hasattr(args, 'chains'):
        config_manager.model_config.n_chains = args.chains
    if hasattr(args, 'target_accept'):
        config_manager.model_config.target_accept = args.target_accept
    
    # Validate configuration if needed
    if not _validate_config(config_manager, args):
        sys.exit(1)
    
    return config_manager

def _validate_config(config_manager, args):
    """
    Validate configuration settings.
    
    Args:
        config_manager: ConfigManager instance
        args: Command line arguments
        
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check if validation is needed
    if args.test_mode:
        # Skip validation in test mode
        logger.info("Skipping configuration validation in test mode")
        return True
    
    if args.force:
        # Skip validation if forcing
        logger.warning("Skipping configuration validation because --force was used")
        return True
    
    # Check validation implementation
    if not hasattr(config_manager, 'validate'):
        logger.warning("Configuration manager does not have a validate method")
        return True
    
    # Execute validation
    try:
        if config_manager.validate():
            return True
        else:
            logger.error("Invalid configuration. Please check your parameters.")
            return False
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return False

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Retail Price Elasticity Analysis")
    
    # General options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--results-dir", type=str, 
                      help="Directory to store results")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      default="INFO", help="Log level")
    parser.add_argument("--run", choices=["full", "diagnostics", "compare", "summarize", "visualize"], 
                      default="full", help="Operation to perform")
    parser.add_argument("--visualize-only", action="store_true",
                      help="Only generate visualizations from existing results")
    
    # Data options
    parser.add_argument("--data-path", type=str, default="data/sales.parquet",
                        help="Path to data file")
    parser.add_argument("--sample-frac", type=float, default=0.1,
                        help="Fraction of data to sample")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with test data")
    
    # Model options
    parser.add_argument("--use-seasonality", action="store_true",
                        help="Include seasonality effects")
    parser.add_argument("--draws", type=int, default=1000,
                        help="Number of draws for MCMC sampling")
    parser.add_argument("--tune", type=int, default=500,
                        help="Number of tuning steps for MCMC sampling")
    parser.add_argument("--chains", type=int, default=2,
                        help="Number of chains for MCMC sampling")
    parser.add_argument("--target-accept", type=float, default=0.8,
                        help="Target acceptance rate for MCMC sampling")
    
    # Output options
    parser.add_argument("--model-type", type=str, default="bayesian",
                        choices=["bayesian"],
                        help="Type of model to use")
    
    # Add force flag to skip validation 
    parser.add_argument("--force", action="store_true",
                      help="Force run even if configuration is invalid")
    
    return parser.parse_args()

def setup_logging(log_level):
    """
    Set up logging based on the specified log level.
    
    Args:
        log_level: Log level to set up
    """
    LoggingManager.setup_logging(
        logger_name="Retail_Elasticity",
        log_level=log_level
    )

if __name__ == "__main__":
    sys.exit(main())
