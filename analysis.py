#!/usr/bin/env python3
"""
Retail Price Elasticity Analysis Functions Library

This module provides specialized functions for retail price elasticity analysis,
with each function having a single responsibility following SOLID principles.
"""
import os
import sys
import logging
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple, Literal, Type, Protocol
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time

# Create a context manager for BLAS warnings instead of global suppression
@contextlib.contextmanager
def suppress_pytensor_warnings():
    """Context manager to temporarily suppress PyTensor BLAS warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='pytensor.tensor.blas')
        yield

# Local imports
from utils.logging_utils import get_logger, LoggingManager, log_step
from model.bayesian_model import BayesianModel
from data.data_loader import DataLoader
from model.model_runner import ModelRunner
from utils.common import ensure_dir_exists, save_json, flatten_dict
from config.config_manager import ConfigManager, get_config
from utils.decorators import log_errors, timed
from utils.analysis.visualizers import (
    visualize_elasticities, 
    create_diagnostic_plots,
    create_model_comparison_plots
)
from utils.results_manager import save_results
from utils.analysis.results_processor import create_comparison_data
from model.exceptions import ModelError

# Get logger for this module
logger = get_logger()

# Optional imports for visualization
try:
    import arviz as az
    import pymc as pm
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    logger.warning("ArviZ not available. Some diagnostic features will be limited.")

# Analysis exceptions
class AnalysisError(Exception):
    """Exception raised for errors in the analysis process."""
    pass

# Data Management Functions
@log_errors(msg="Error loading analysis data", reraise=False)
def load_analysis_data(config) -> pd.DataFrame:
    """
    Load and prepare data for analysis.
    
    Args:
        config: Configuration object with data_config attribute
        
    Returns:
        Prepared data as DataFrame
        
    Raises:
        AnalysisError: If data loading or preparation fails
    """
    try:
        logger.info(f"Loading data from {config.data_config.data_path}")
        
        # Create direct data loader
        loader = DataLoader(
            data_path=config.data_config.data_path,
            price_col=config.data_config.price_col,
            quantity_col=config.data_config.quantity_col,
            sku_col=config.data_config.sku_col,
            product_class_col=config.data_config.product_class_col,
            date_col=config.data_config.date_col
        )
        
        # Special case for test data - set column mappings directly
        if hasattr(config.data_config, 'column_mappings') and config.data_config.column_mappings:
            logger.info(f"Setting column mappings: {config.data_config.column_mappings}")
            loader.set_column_mapping(config.data_config.column_mappings)
        
        # Load the data
        data = loader.load_data(
            sample_frac=config.data_config.sample_frac,
            log_transform_price=config.data_config.log_transform_price,
            log_transform_quantity=config.data_config.log_transform_quantity,
            add_date_features=True
        )
        
        # Check if we have data
        if data.empty:
            raise AnalysisError("Empty dataset after loading. Check your filters.")            
        return data
        
    except Exception as e:
        raise AnalysisError(f"Error loading analysis data: {str(e)}") from e

@log_errors(msg="Error filtering data", reraise=False)
def filter_data(data: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """Apply filtering conditions to the data."""
    filtered_data = data.copy()
    
    for column, value in conditions.items():
        if column in filtered_data.columns:
            if isinstance(value, (list, tuple)):
                filtered_data = filtered_data[filtered_data[column].isin(value)]
            else:
                filtered_data = filtered_data[filtered_data[column] == value]
                
    logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} rows")
    return filtered_data

# Model interface definition
class ElasticityModel(Protocol):
    """Protocol for elasticity models."""
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Fit model to data."""
        ...
    
    def summarize(self) -> Dict[str, Any]:
        """Get model summary."""
        ...

@dataclass
class ModelResults:
    """Structured container for model results"""
    elasticities: Optional[Dict[str, Any]] = None
    trace: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class ComparisonResults:
    """Structured container for model comparison results."""
    seasonal_model: ModelResults
    nonseasonal_model: ModelResults
    metrics_diff: Dict[str, float] = field(default_factory=dict)
    waic_comparison: Optional[Dict[str, float]] = None
    status: str = "success"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "seasonal_model": self.seasonal_model.to_dict(),
            "nonseasonal_model": self.nonseasonal_model.to_dict(),
            "metrics_diff": self.metrics_diff,
            "waic_comparison": self.waic_comparison,
            "status": self.status
        }

@dataclass
class ModelParameters:
    """Parameter object for model configuration."""
    data_path: str
    results_dir: str
    model_type: str = "bayesian"
    use_seasonality: bool = False
    sample_frac: float = 0.1
    n_draws: int = 1000
    n_tune: int = 500
    n_chains: int = 2
    target_accept: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class ModelFactory:
    """Factory for creating elasticity models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        use_seasonality: bool,
        elasticity_prior_mean: Optional[float] = None,
        elasticity_prior_std: Optional[float] = None,
        class_effect_std: Optional[float] = None,
        sku_effect_std: Optional[float] = None
    ) -> ElasticityModel:
        """Create an elasticity model instance."""
        config = get_config()
        
        # Use provided values or defaults from config
        elasticity_prior_mean = elasticity_prior_mean if elasticity_prior_mean is not None else config.elasticity_prior_mean
        elasticity_prior_std = elasticity_prior_std if elasticity_prior_std is not None else config.elasticity_prior_std
        class_effect_std = class_effect_std if class_effect_std is not None else config.class_effect_std
        sku_effect_std = sku_effect_std if sku_effect_std is not None else config.sku_effect_std
        
        if model_type == "bayesian":
            # Create results directory for the model
            results_dir = Path("results") / f"model_{int(time.time())}"
            
            # Create model config with the prior parameters
            model_config = {
                "elasticity_prior_mean": elasticity_prior_mean,
                "elasticity_prior_std": elasticity_prior_std,
                "class_effect_std": class_effect_std,
                "sku_effect_std": sku_effect_std
            }
            
            model = BayesianModel(
                results_dir=results_dir,
                model_name="bayesian_elasticity",
                use_seasonality=use_seasonality,
                model_config=model_config,
                n_draws=1000,
                n_tune=500,
                n_chains=2,
                target_accept=0.8
            )
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# Model Execution Functions
@log_errors(msg="Error running model", reraise=False)
def run_model(
    model: ElasticityModel, 
    data: pd.DataFrame,
    n_draws: int = 1000,
    n_tune: int = 500,
    n_chains: int = 2,
    target_accept: float = 0.8
) -> ModelResults:
    """Run the model with the specified data and sampling parameters."""
    logger.info("Running model with parameters: " +
               f"draws={n_draws}, tune={n_tune}, chains={n_chains}")
    
    # Prepare sampling parameters
    sampling_params = {
        "n_draws": n_draws,
        "n_tune": n_tune,
        "n_chains": n_chains,
        "target_accept": target_accept
    }
    
    # Fit the model and return results
    raw_results = model.fit(data, **sampling_params)
    
    # Convert to ModelResults
    results = ModelResults(
        elasticities=raw_results.get('elasticities'),
        trace=raw_results.get('trace'),
        metrics=raw_results.get('metrics', {}),
        diagnostics=raw_results.get('diagnostics', {})
    )
    
    return results

# Core analysis workflow function - used by multiple analysis methods
@timed
@log_errors(msg="Error running core analysis", reraise=False)
def _run_core_analysis(params: ModelParameters) -> Tuple[pd.DataFrame, ElasticityModel, ModelResults]:
    """Core function to run common analysis steps and return data, model and results."""
    # 1. Load data
    data = load_analysis_data(params.data_path)
    
    # 2. Create model
    model = ModelFactory.create_model(
        model_type=params.model_type,
        use_seasonality=params.use_seasonality
    )
    
    # 3. Run model
    results = run_model(
        model=model,
        data=data,
        n_draws=params.n_draws,
        n_tune=params.n_tune,
        n_chains=params.n_chains,
        target_accept=params.target_accept
    )

    return data, model, results

# Analysis Orchestration Functions
@timed("Full analysis")
@log_errors(AnalysisError, msg="Error running full analysis", reraise=False)
def run_full_analysis(
    config_manager,
    results_dir="results",
    debug=False
) -> dict:
    """
    Run a complete elasticity analysis pipeline with automatic model selection.
    
    Args:
        config_manager: Configuration manager object
        results_dir: Directory to store results
        debug: Whether to run in debug mode
        
    Returns:
        Dictionary of results
    """
    logger.info("Starting full elasticity analysis...")
    
    try:
        # Load data using config
        data = load_analysis_data(config_manager)
        
        # Create model factory
        from model.model_runner import ModelFactory
        model_factory = ModelFactory()
        
        # Create model
        model = model_factory.create_model(
            config_manager.model_config.model_type,
            model_config={
                "use_seasonality": config_manager.model_config.use_seasonality,
                "elasticity_prior_mean": config_manager.model_config.elasticity_prior_mean,
                "elasticity_prior_std": config_manager.model_config.elasticity_prior_std,
                "class_effect_std": config_manager.model_config.class_effect_std,
                "sku_effect_std": config_manager.model_config.sku_effect_std
            }
        )
        
        # Set up sampling parameters
        sampling_params = {
            "n_draws": config_manager.model_config.n_draws,
            "n_tune": config_manager.model_config.n_tune,
            "n_chains": config_manager.model_config.n_chains,
            "target_accept": config_manager.model_config.target_accept,
        }
        
        # Run model
        logger.info(f"Running model with parameters: " + 
                   f"draws={sampling_params['n_draws']}, " +
                   f"tune={sampling_params['n_tune']}, " +
                   f"chains={sampling_params['n_chains']}")
        
        results = model.fit(data, **sampling_params)
        
        # Return results
        return results
        
    except Exception as e:
        # If any step fails, log and propagate
        logger.error(f"Analysis failed: {str(e)}")
        raise AnalysisError(f"Full analysis failed: {str(e)}") from e

# Executes Model Diagnostics
@timed
@log_errors(msg="Error running diagnostics", reraise=False)
def run_diagnostics(
    data_path: str,
    results_dir: str,
    model_type: str = "bayesian",
    use_seasonality: bool = False,
    sample_frac: float = 0.1,
    n_draws: int = 1000,
    n_tune: int = 500,
    n_chains: int = 2,
    target_accept: float = 0.8
) -> None:
    """Run model diagnostics to evaluate convergence and fit."""
    logger.info("Running model diagnostics...")
    
    # Create parameter object
    params = ModelParameters(
        data_path=data_path,
        results_dir=results_dir,
        model_type=model_type,
        use_seasonality=use_seasonality,
        sample_frac=sample_frac,
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept
    )
    
    # Run core analysis to get data, model and results
    data, model, results = _run_core_analysis(params)
    
    # Create diagnostic visualizations
    if hasattr(model, 'create_diagnostics') and callable(getattr(model, 'create_diagnostics')):
        model.create_diagnostics(results.trace)
        
    # Create diagnostic plots
    create_diagnostic_plots(
        results.trace, 
        os.path.join(results_dir, "diagnostics"),
        model_type=model_type,
        use_seasonality=use_seasonality
    )
    
    # Save diagnostics
    save_results(
        results.diagnostics, 
        results_dir, 
        result_type="diagnostics",
        filename_prefix=f"{model_type}_{'seasonal' if use_seasonality else 'nonseasonal'}"
    )
    
    logger.info("Diagnostics completed successfully")

# Compares Models with and without seasonality
@timed
@log_errors(msg="Error comparing models", reraise=False)
def compare_models(
    data_path: str, 
    results_dir: str,
    model_type: str = "bayesian",
    sample_frac: float = 0.1,
    n_draws: int = 1000,
    n_tune: int = 500,
    n_chains: int = 2,
    target_accept: float = 0.8
) -> ComparisonResults:
    """Compare models with and without seasonality."""
    logger.info("Running model comparison...")
    
    # 1. Run seasonal model
    _, _, seasonal_results = _run_core_analysis(
        data_path=data_path,
        results_dir=results_dir,
        model_type=model_type,
        use_seasonality=True,
        sample_frac=sample_frac,
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept
    )
    
    # 2. Run non-seasonal model
    _, _, nonseasonal_results = _run_core_analysis(
        data_path=data_path,
        results_dir=results_dir,
        model_type=model_type,
        use_seasonality=False,
        sample_frac=sample_frac,
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept
    )
    
    # 3. Create comparison data
    raw_comparison_data = create_comparison_data(
        seasonal_results=seasonal_results.to_dict(),
        nonseasonal_results=nonseasonal_results.to_dict()
    )
    
    # 4. Create structured comparison results
    comparison_results = ComparisonResults(
        seasonal_model=seasonal_results,
        nonseasonal_model=nonseasonal_results,
        metrics_diff=raw_comparison_data.get('metrics_diff', {}),
        waic_comparison=raw_comparison_data.get('waic_comparison')
    )
    
    # 5. Save comparison results using consolidated function
    save_results(comparison_results.to_dict(), results_dir, 
               result_type="comparison", filename_prefix=f"{model_type}_comparison")
    
    # 6. Create comparison visualizations
    create_model_comparison_plots(
        seasonal_results=seasonal_results.to_dict(),
        nonseasonal_results=nonseasonal_results.to_dict(),
        output_dir=Path(results_dir) / "comparisons"
    )
    
    return comparison_results


@dataclass
class ModelFeatures:
    """Structured container for model feature information."""
    features: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "bayesian"
    use_seasonality: bool = False
    status: str = "success"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


def summarize_model_features(
    results_dir: str,
    model_type: str = "bayesian",
    use_seasonality: bool = False
) -> ModelFeatures:
    """Summarize model features and parameters."""
    model = ModelFactory.create_model(
        model_type=model_type,
        use_seasonality=use_seasonality
    )
    
    # Get model feature summary
    feature_summary = model.summarize() if hasattr(model, 'summarize') else {}
    
    # Create structured result
    model_features = ModelFeatures(
        features=feature_summary,
        model_type=model_type,
        use_seasonality=use_seasonality
    )
    
    # Save summary
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "model_features.json", 'w') as f:
        import json
        json.dump(model_features.to_dict(), f, indent=4)
    
    logger.info(f"Saved model feature summary to {output_dir}")
    
    return model_features 