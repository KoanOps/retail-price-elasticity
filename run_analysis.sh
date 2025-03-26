#!/bin/bash
# ==============================================================================
# Retail Price Elasticity Analysis Runner
# ==============================================================================
#
# PURPOSE:
# This script provides a convenient way to execute the price elasticity analysis
# with proper Python path configuration. It ensures the application can find all
# required modules regardless of the current working directory.
#
# USAGE:
#   ./run_analysis.sh [options]
#
# COMMON OPTIONS:
#   --model-type bayesian     Use Bayesian hierarchical model
#   --data-path data/sales.parquet     Path to retail sales data
#   --sample 0.3              Use 30% of data for faster analysis
#   --results-dir results/analysis1    Directory for results
#   --draws 1000              Number of MCMC draws (Bayesian model)
#   --tune 500                Number of tuning iterations (Bayesian model)
#
# EXAMPLES:
#   ./run_analysis.sh --model-type bayesian --data-path data/sales.parquet --results-dir results/my_analysis
#   ./run_analysis.sh --help    Show all available options
#
# NOTES:
#   - Requires Python 3.7+ with all dependencies installed
#   - See requirements.txt for required packages
#   - Results will be saved to the specified directory
# ==============================================================================

# Add the project root to PYTHONPATH to ensure all imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the analysis script with all arguments passed through
python analysis.py "$@"
echo "Analysis complete. Check results directory for outputs." 