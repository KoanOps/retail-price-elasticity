# Retail Price Elasticity Framework

A comprehensive toolkit for analyzing price elasticity in retail data using Bayesian hierarchical models. This framework provides data preprocessing, model estimation, validation, and visualization capabilities for retail price optimization.

## About Price Elasticity

Price elasticity measures how demand responds to price changes. 
- Elasticity < -1: Elastic (price-sensitive)
- Elasticity = -1: Unit elastic
- Elasticity > -1: Inelastic (less price-sensitive)

## Model Philosophy
- Materially right > Precisely wrong
- Signal + actionability is underrated, model precision is overrated
- Design for reliable decision-making under uncertainty, not surgical price optimization
- Different markets have fundamentally different data-generating processes

## Data Characteristics
- **Sparse SKU Data**: Hierarchical structure manages varying observation counts of unique price points per SKU with product class effects (categorical data)
- **Long-Tailed Distributions**: Log transformations handle skewed price and quantity distributions
- **Correlated Patterns**: Partial pooling captures relationships between SKUs and product classes
- **Temporal Effects**: Seasonal components model time-based demand patterns

## Conceptual Flow
Raw Data → Data Preprocessing → Hierarchical Bayesian Model → Posterior Analysis → Visualizations/Reports
```

## Repository Structure

```
retail/
├── data/                  # Data management components
│   ├── data_loader.py                # Loads and validates input data
│   ├── data_preprocessor.py          # Prepares data for modeling
│   ├── data_visualizer.py            # Visualization utilities
│   ├── simulation.py                 # Synthetic data generation
│   ├── generate_data.py              # Data generation utilities
│   └── sales.parquet                 # Sample retail sales data
│
├── model/                 # Elasticity modeling components
│   ├── bayesian/                     # Bayesian modeling components 
│   │   ├── model_builder.py          # PyMC model construction
│   │   └── __init__.py               # Package exports
│   │
│   ├── base_model.py                 # Abstract base model class
│   ├── bayesian_model.py             # Main Bayesian model implementation
│   ├── linear_model.py               # Linear elasticity model
│   ├── model_runner.py               # Model execution orchestration
│   ├── data_preparation.py           # Data preparation for models
│   ├── constants.py                  # Model constants and defaults
│   ├── sampling.py                   # MCMC sampling management
│   ├── visualization.py              # Visualization utilities
│   ├── diagnostics.py                # Model diagnostic tools
│   └── exceptions.py                 # Custom exception classes
│
├── utils/                  # Utility functions and helpers
│   ├── analysis/                     # Analysis utilities
│   │   ├── model_validation.py       # Validation tools
│   │   ├── log_transform_validator.py # Log transformation utility
│   │   ├── seasonality_analysis.py   # Seasonality analysis tools
│   │   └── results_processor.py      # Results processing utilities
│   │
│   ├── results_manager.py            # Results processing and storage
│   ├── logging_utils.py              # Enhanced logging
│   ├── decorators.py                 # Function decorators
│   ├── common.py                     # Common utility functions
│   └── data_utils.py                 # Data manipulation utilities
│
├── tests/                  # Test and validation suite
│   └── test_*.py                     # Unit tests for components
│
├── config/                 # Configuration management
│   └── default_config.py             # Default parameter settings
│
├── diagnostics/            # Additional diagnostic tools
├── scripts/                # Helper scripts
├── results/                # Output directory for analysis results
├── validation_results/     # Validation output directory
├── analysis.py             # Main analysis script
├── main.py                 # Command-line entry point
├── run_analysis.sh # Shell script for running analysis
└── requirements.txt        # Project dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/KoanOps/retail-price-elasticity.git
cd retail-price-elasticity

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package (optional, for command-line use)
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line interface through the `retail-analysis` command (when installed) or by running `main.py` directly:

```bash
# If installed as a package
retail-analysis --data-path data/sales.parquet --results-dir results/my_analysis --run full

# If running directly
python main.py --data-path data/sales.parquet --results-dir results/my_analysis --run full
```

Available options:

```bash
python main.py --help
```

Common parameters:
```bash
python main.py \
  --data-path data/sales.parquet \
  --results-dir results/cli_run \
  --model-type bayesian \
  --sample-frac 0.3 \
  --draws 1000 \
  --tune 500 \
  --chains 2
```

### Model Validation

To validate model accuracy using synthetic data:

```bash
python tests/validate_elasticity_model.py \
  --model-type bayesian \
  --observations 10000 \
  --skus 50 \
  --sample-frac 0.3 \
  --draws 1000 \
  --tune 500 \
  --results-dir results/validation_test
```

## Utility Tools

### Log Transform Validator

The framework includes a dedicated utility to verify that log transformation is appropriate for your price and quantity variables:

```bash
# Check if log transformation is appropriate for your data
python utils/analysis/log_transform_validator.py --data-path data/sales.parquet

# Test with synthetic distributions
python utils/analysis/log_transform_validator.py --test-synthetic
```

The utility evaluates three key metrics to determine appropriateness:
- Skewness comparison between raw and log-transformed data
- Shapiro-Wilk normality test
- Q-Q plot correlation coefficients

## Model Features

- **Hierarchical Structure**: Balances between "complete pooling" (all SKUs have identical elasticity) and "no pooling" (each SKU is fully independent). Uses partial pooling to optimize bias-variance trade-off.
- **Uncertainty Quantification**: Provides credible intervals around elasticity estimates, allowing informed decisions under uncertainty.
- **Diagnostics and Validation**:
  - MCMC convergence checks
  - Posterior predictive checks
  - Validation on synthetic data with known elasticities
- **Performance Optimization**:
  - Non-centered parameterization to handle hierarchical complexity
  - Optimized MCMC sampling (target_accept=0.95, adaptive tuning, multiple chains) suitable for large retail datasets

## Model Assumptions
- **Demand Functional Form**: Assumes a log-log demand relationship (constant elasticity model).
- **Prior Distributions**: Elasticity priors are normally distributed, centered around -1.0, adjustable via configuration parameters.
- **Observation Noise**: Currently modeled with a HalfNormal distribution. Inverse gamma could replace this if historical data suggests known variance patterns.
- **Cross-Price Effects**: Model currently focuses only on own-price elasticity; explicit modeling of cross-price or promotional interaction terms is not implemented.
- **Exchangeability**: Assumes SKUs within a product class are exchangeable, allowing hierarchical parameters to generalize.
- **Customization**: model_config parameters allow adjusting priors, pooling strength, and elasticity bounds for market-specific adaptations (luxury vs. necessities).

## Limitations
- **Data Requirements**: Currently requires clean, structured price-quantity data
- **Computation Time**: Full Bayesian inference can be computationally intensive for very large datasets. 
- **External Factors**: Current model doesn't account for all external factors (i.e.competitor pricing)
- **Regime Stability**: Assumes stable price-demand relationships across time periods; phase shifts (e.g., COVID, inflation spikes) require detecting regime changes and adapting the modeling framework accordingly

## Future work
Future enhancements could include:
- Adding additional external factors i.e. competitor price effects
- Room to adapt model in different regimes/phase shifts
- Standardizing data to improve performance and efficiency
- Optimizing hierarchical model parameterization and MCMC tuning