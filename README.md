# Retail Price Elasticity Framework

A comprehensive toolkit for analyzing price elasticity in retail data using Bayesian hierarchical models. This framework provides data preprocessing, model estimation, validation, and visualization capabilities for retail price optimization.

## About Price Elasticity

Price elasticity measures how demand responds to price changes. 
- Elasticity < -1: Elastic (price-sensitive)
- Elasticity = -1: Unit elastic
- Elasticity > -1: Inelastic (less price-sensitive)

This Bayesian approach provides robust elasticity estimates with uncertainty quantification, accounting for:
- SKU-level differences
- Product category effects
- Seasonality patterns
- Price interactions

## Repository Structure

```
retail/
├── data/                  # Data management components
│   ├── data_loader.py                # Loads and validates input data
│   ├── data_preprocessor.py          # Prepares data for modeling
│   ├── data_visualizer.py            # Visualization utilities
│   └── simulation.py                 # Synthetic data generation
│
├── model/                 # Elasticity modeling components
│   ├── bayesian/                     # Bayesian modeling components 
│   │   ├── data_preparation.py       # Bayesian-specific preprocessing
│   │   ├── model_builder.py          # PyMC model construction
│   │   ├── sampling.py               # MCMC sampling management
│   │   └── visualization.py          # Model-specific visualization
│   │
│   ├── bayesian_model.py             # Main Bayesian model implementation
│   └── model_runner.py               # Model execution orchestration
│
├── utils/                  # Utility functions and helpers
│   ├── model_validation.py           # Validation tools
│   ├── results_manager.py            # Results processing and storage
│   ├── logging_utils.py              # Enhanced logging
│   ├── decorators.py                 # Function decorators
│   └── log_transform_validator.py    # Log transformation utility
│
├── tests/                  # Test and validation suite
│   ├── validate_elasticity_model.py  # Model validation script
│   └── test_*.py                     # Unit tests for components
│
├── config/                 # Configuration management
│   └── default_config.py             # Default parameter settings
│
├── results/                # Output directory for analysis results
├── analysis.py                       # Main analysis script
├── main.py                           # Command-line entry point
└── requirements.txt                  # Project dependencies
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

### Python API

You can integrate the framework into your Python applications:

```python
# Import the model runner
from model.model_runner import ModelRunner
from config.config_manager import ConfigManager

# Initialize configuration
config_manager = ConfigManager()
config_manager.data_config.data_path = "data/sales.parquet"
config_manager.app_config.results_dir = "results/api_run"

# Initialize model runner with configuration
runner = ModelRunner(
    results_dir=config_manager.app_config.results_dir,
    config_manager=config_manager
)

# Run analysis
model_params = {
    'model_type': 'bayesian',
    'sample_frac': 0.3,
    'use_seasonality': True,
    'n_draws': 1000,
    'n_tune': 500,
    'n_chains': 2
}
results = runner.run_analysis(config_manager.data_config.data_path, **model_params)

# Access elasticities (check actual structure in your results)
if "elasticities" in results:
    elasticities = results["elasticities"]
    print(f"Elasticity results available for {len(elasticities)} SKUs")
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

- **Hierarchical Structure**: Accounts for SKU and category-level effects
- **Prior Specifications**: Informed priors based on retail domain knowledge
- **Diagnostics**: MCMC convergence checks and posterior predictive checks
- **Validation**: Synthetic data validation with known elasticities
- **Uncertainty Quantification**: Credible intervals for all estimates
- **Performance**: Optimized for large retail datasets

## Limitations
- **Data Requirements**: Currently requires clean, structured price-quantity data
- **Computation Time**: Full Bayesian inference can be computationally intensive for very large datasets
- **External Factors**: Current model doesn't account for all external factors (i.e.competitor pricing)

## Future work
Future enhancements could include:
- Adding competitor price effects
- Implementing multi-SKU cross-elasticity
- Developing a web dashboard for non-technical users