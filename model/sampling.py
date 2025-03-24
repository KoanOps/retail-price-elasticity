"""
Bayesian model sampling component.

This module provides functionality for MCMC sampling from PyMC models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, cast
from pathlib import Path
import logging
import os

from utils.logging_utils import logger, log_step
from model.exceptions import SamplingError
from model.constants import (
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    DEFAULT_CHAINS,
    DEFAULT_TARGET_ACCEPT,
    ELASTIC_THRESHOLD
)


# In production, these imports come from dependency injection
# For development/type checking only
try:
    import pymc as _pm_type_hint  # For type hints only
    import arviz as _az_type_hint  # For type hints only
except ImportError:
    pass


class BayesianSampler:
    """
    Handles MCMC sampling for Bayesian elasticity models.
    
    This component is responsible for:
    - Running MCMC sampling algorithms
    - Trace processing and saving
    - Diagnostics calculation
    """
    
    def __init__(
        self, 
        pymc=None,
        n_draws: int = 1000,
        n_tune: int = 500,
        n_chains: int = 2,
        target_accept: float = 0.95,
        max_treedepth: int = 15,
        results_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the sampler.
        
        Args:
            pymc: PyMC module (injected at runtime)
            n_draws: Number of sampling draws after tuning
            n_tune: Number of tuning steps
            n_chains: Number of MCMC chains to run
            target_accept: Target acceptance rate for NUTS sampler
            max_treedepth: Maximum depth of the binary tree used by NUTS
            results_dir: Directory to save results to
        """
        self.pymc = pymc
        if self.pymc is None:
            logger.warning("PyMC not available. Sampling functionality will be limited.")
            
        # MCMC parameters
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        
        # Results
        self.results_dir = Path(results_dir) if results_dir else None
        self.trace = None
        self.summary = None
    
    @log_step("Running MCMC sampling")
    def sample(self, model: Any) -> Any:
        """
        Run MCMC sampling on the given PyMC model.
        
        Args:
            model: PyMC model
            
        Returns:
            MCMC trace
            
        Raises:
            SamplingError: If sampling fails
        """
        try:
            pymc = self.pymc
            if pymc is None:
                raise SamplingError("PyMC not available")
                
            logger.info(f"Starting MCMC sampling with parameters: draws={self.n_draws}, tune={self.n_tune}, chains={self.n_chains}, target_accept={self.target_accept}, max_treedepth={self.max_treedepth}")
            
            # Run MCMC sampling
            # The model should already be a PyMC model context
            with model:
                trace = pymc.sample(
                    draws=self.n_draws,
                    tune=self.n_tune,
                    chains=self.n_chains,
                    target_accept=self.target_accept,
                    return_inferencedata=True
                )
            
            # Store the trace
            self.trace = trace
            
            # Create summary (simple version without ArviZ)
            self.summary = {
                'n_samples': self.n_draws * self.n_chains,
                'n_tune': self.n_tune,
                'n_chains': self.n_chains,
                'target_accept': self.target_accept,
                'max_treedepth': self.max_treedepth
            }
            
            logger.info(f"Completed MCMC sampling with {self.summary['n_samples']} samples")
            
            return trace
            
        except ValueError as e:
            raise SamplingError(f"Invalid parameter for MCMC sampling: {str(e)}")
        except RuntimeError as e:
            raise SamplingError(f"Runtime error during MCMC sampling: {str(e)}")
        except TypeError as e:
            raise SamplingError(f"Type error during MCMC sampling: {str(e)}")
        except Exception as e:
            # Fall back to generic exception only if specific exceptions don't apply
            raise SamplingError(f"Unexpected error during MCMC sampling: {str(e)}")
    
    def save_trace(self) -> Path:
        """
        Save the sampling trace to disk.
        
        Returns:
            Path to saved trace file
            
        Raises:
            SamplingError: If trace cannot be saved
        """
        if self.trace is None:
            raise SamplingError("No trace available to save. Run sampling first.")
            
        if self.arviz is None:
            logger.warning("ArviZ not available. Cannot save trace in preferred format.")
            return Path()
            
        try:
            # Create output directory
            trace_dir = self.results_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trace using ArviZ
            trace_path = trace_dir / "trace.nc"
            self.arviz.to_netcdf(self.trace, str(trace_path))
            
            logger.info(f"Saved sampling trace to {trace_path}")
            
            return trace_path
            
        except FileNotFoundError as e:
            raise SamplingError(f"Directory not found when saving trace: {str(e)}")
        except PermissionError as e:
            raise SamplingError(f"Permission denied when saving trace: {str(e)}")
        except IOError as e:
            raise SamplingError(f"I/O error when saving trace: {str(e)}")
        except Exception as e:
            # Fall back to generic exception only if specific exceptions don't apply
            raise SamplingError(f"Unexpected error saving trace: {str(e)}")
    
    def extract_elasticities(
        self, 
        sku_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract elasticity estimates from the trace.
        
        Args:
            sku_names: List of SKU names/identifiers
            class_names: List of product class names/identifiers
            
        Returns:
            DataFrame with elasticity estimates
            
        Raises:
            SamplingError: If elasticities cannot be extracted
        """
        if self.trace is None:
            raise SamplingError("No trace available. Run sampling first.")
            
        if self.arviz is None:
            raise SamplingError("ArviZ not available. Cannot extract elasticities in standard way.")
            
        try:
            # For readability
            az = self.arviz
            
            # Extract SKU-level elasticities
            sku_elasticity = az.summary(self.trace, var_names=["sku_elasticity"])
            
            # Extract class-level elasticities
            class_elasticity = az.summary(self.trace, var_names=["class_elasticity"])
            
            # Extract global elasticity
            global_elasticity = az.summary(self.trace, var_names=["global_elasticity"])
            
            # Create DataFrame for SKU-level elasticities
            sku_df = pd.DataFrame({
                'sku_id': range(len(sku_elasticity)),
                'mean': sku_elasticity['mean'].values,
                'sd': sku_elasticity['sd'].values,
                'hdi_3%': sku_elasticity['hdi_3%'].values,
                'hdi_97%': sku_elasticity['hdi_97%'].values
            })
            
            # Add SKU names if provided
            if sku_names is not None:
                sku_df['sku_name'] = sku_names
                
            # Add product class information if available
            if class_names is not None and hasattr(self.trace, 'product_class_idx'):
                # Use product_class_idx from trace
                product_class_idx = self.trace.product_class_idx
                
                # Create mapping array from SKU to product class
                sku_to_class = []
                for sku_id in range(len(sku_names)):
                    class_id = product_class_idx[sku_id]
                    class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                    sku_to_class.append(class_name)
                    
                # Add product class to DataFrame
                sku_df['product_class'] = sku_to_class
                
                # Add class-level elasticity information
                class_to_elasticity = {
                    class_names[i]: class_elasticity.iloc[i]['mean'] 
                    for i in range(len(class_names))
                }
                sku_df['class_elasticity'] = sku_df['product_class'].map(class_to_elasticity)
            elif class_names is not None:
                logger.warning("Product class mapping not available in trace data")
                
            # Add elasticity interpretation using constant threshold
            sku_df['elastic'] = sku_df['mean'] < ELASTIC_THRESHOLD
            
            logger.info(f"Extracted elasticity estimates for {len(sku_df)} SKUs")
            
            return sku_df
            
        except KeyError as e:
            raise SamplingError(f"Missing variable in trace: {str(e)}")
        except IndexError as e:
            raise SamplingError(f"Index error extracting elasticities: {str(e)}")
        except ValueError as e:
            raise SamplingError(f"Value error extracting elasticities: {str(e)}")
        except AttributeError as e:
            raise SamplingError(f"Attribute error extracting elasticities: {str(e)}")
        except Exception as e:
            # Fall back to generic exception only if specific exceptions don't apply
            raise SamplingError(f"Unexpected error extracting elasticities: {str(e)}")
            
    def get_trace(self) -> Any:
        """
        Get the sampling trace.
        
        Returns:
            ArviZ InferenceData or PyMC trace object or None if not available
        """
        return self.trace
        
    def get_sampling_stats(self) -> Dict[str, Any]:
        """
        Get sampling statistics.
        
        Returns:
            Dictionary with sampling statistics
        """
        return self.summary.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the MCMC sampling.
        
        Returns:
            Dictionary with summary statistics
            
        Raises:
            SamplingError: If summary not available
        """
        if not hasattr(self, 'summary') or self.summary is None:
            raise SamplingError("No sampling summary available. Run sampling first.")
            
        return self.summary 