"""
Diagnostics module for Bayesian elasticity models.

This module provides diagnostic tools to evaluate the quality of
Bayesian elasticity models and MCMC sampling.
"""
from typing import Dict, List, Optional, Any, Tuple, Union, cast
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from utils.logging_utils import logger, log_step
from model.exceptions import VisualizationError

# Type imports for PyMC and ArviZ
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning("PyMC not available. Bayesian diagnostics will be disabled.")


class BayesianDiagnostics:
    """
    Provides diagnostics for Bayesian elasticity models.
    
    Responsibilities:
    - Computing convergence diagnostics
    - Assessing model fit
    - Producing diagnostic plots
    - Evaluating predictive accuracy
    """
    
    def __init__(
        self,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize the diagnostics component.
        
        Args:
            results_dir: Directory to save diagnostic plots
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC and ArviZ are required for Bayesian diagnostics")
            
        self.results_dir = results_dir
        if self.results_dir is not None:
            self.diagnostics_dir = self.results_dir / "diagnostics"
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.diagnostics_dir = None
            
    def compute_diagnostics(self, trace: "az.InferenceData") -> Dict[str, Any]:
        """
        Compute diagnostic metrics for the trace.
        
        Args:
            trace: ArviZ InferenceData object with posterior samples
            
        Returns:
            Dictionary of diagnostic metrics
            
        Raises:
            VisualizationError: If computation fails
        """
        try:
            # Compute MCMC diagnostics
            summary = az.summary(trace, round_to=3)
            
            # Extract Rhat statistics
            rhat_values = summary["r_hat"].values
            n_divergent = trace.sample_stats.diverging.sum().item()
            
            # Calculate MCSE / SD ratios (should be < 0.05)
            mcse_sd_ratios = summary["mcse"] / summary["sd"]
            
            # Compute diagnostics summary
            diagnostics = {
                "n_eff_min": summary["ess_bulk"].min(),
                "n_eff_mean": summary["ess_bulk"].mean(),
                "rhat_max": rhat_values.max(),
                "rhat_mean": rhat_values.mean(),
                "n_divergent": n_divergent,
                "mcse_sd_max": mcse_sd_ratios.max(),
                "mcse_sd_mean": mcse_sd_ratios.mean(),
                "n_parameters": len(summary),
                "converged": bool(rhat_values.max() < 1.05 and n_divergent == 0)
            }
            
            logger.info(f"Computed diagnostics: max Rhat = {diagnostics['rhat_max']:.3f}, "
                      f"min ESS = {diagnostics['n_eff_min']:.1f}, "
                      f"n_divergent = {diagnostics['n_divergent']}")
            
            # Save summary table if results directory is specified
            if self.diagnostics_dir is not None:
                summary_path = self.diagnostics_dir / "summary.csv"
                summary.to_csv(summary_path)
                logger.info(f"Saved summary table to {summary_path}")
            
            return diagnostics
            
        except Exception as e:
            raise VisualizationError(f"Diagnostic computation failed: {str(e)}")
            
    def plot_trace(self, 
                  trace: "az.InferenceData", 
                  variables: Optional[List[str]] = None,
                  filename: str = "trace_plot.png") -> Optional[Path]:
        """
        Generate trace plots for model parameters.
        
        Args:
            trace: ArviZ InferenceData object with posterior samples
            variables: List of variables to plot (if None, plots key parameters)
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.diagnostics_dir is None:
            logger.warning("No diagnostics directory specified")
            return None
            
        try:
            # Default to key variables if none specified
            if variables is None:
                variables = ["global_elasticity", "intercept"]
                
                # Add first few elements of array variables if they exist
                if "class_elasticity" in trace.posterior.data_vars:
                    variables.append("class_elasticity")
                    
            # Create the plot
            fig, axes = plt.subplots(len(variables), 2, figsize=(12, 4 * len(variables)))
            axes = az.plot_trace(trace, var_names=variables, axes=axes)
            plt.tight_layout()
            
            # Save the plot
            output_path = self.diagnostics_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved trace plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Trace plotting failed: {str(e)}")
            
    def plot_posterior(self, 
                      trace: "az.InferenceData", 
                      variables: Optional[List[str]] = None,
                      filename: str = "posterior_plot.png") -> Optional[Path]:
        """
        Generate posterior distribution plots.
        
        Args:
            trace: ArviZ InferenceData object with posterior samples
            variables: List of variables to plot (if None, plots key parameters)
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.diagnostics_dir is None:
            logger.warning("No diagnostics directory specified")
            return None
            
        try:
            # Default to key variables if none specified
            if variables is None:
                variables = ["global_elasticity", "intercept"]
                
                # Add class-level elasticities if they exist
                if "class_elasticity" in trace.posterior.data_vars:
                    n_classes = trace.posterior.class_elasticity.shape[-1]
                    if n_classes <= 10:  # Limit to avoid overcrowded plots
                        variables.append("class_elasticity")
                    
            # Create the plot
            fig, axes = plt.subplots(len(variables), 1, figsize=(10, 4 * len(variables)))
            axes = az.plot_posterior(trace, var_names=variables, ax=axes)
            plt.tight_layout()
            
            # Save the plot
            output_path = self.diagnostics_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved posterior plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Posterior plotting failed: {str(e)}")
            
    def plot_forest(self,
                   trace: "az.InferenceData",
                   elasticity_df: pd.DataFrame,
                   top_n: int = 20,
                   filename: str = "elasticity_forest_plot.png") -> Optional[Path]:
        """
        Generate a forest plot of elasticity estimates for top SKUs.
        
        Args:
            trace: ArviZ InferenceData object with posterior samples
            elasticity_df: DataFrame with elasticity estimates
            top_n: Number of SKUs to include in the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.diagnostics_dir is None:
            logger.warning("No diagnostics directory specified")
            return None
            
        try:
            # Select top SKUs by elasticity magnitude
            top_skus = elasticity_df.sort_values("elasticity_mean").head(top_n)
            
            # Create forest plot
            fig, ax = plt.subplots(figsize=(10, top_n * 0.3 + 2))
            
            # Plot confidence intervals
            y_pos = np.arange(len(top_skus))
            ax.errorbar(
                x=top_skus["elasticity_mean"],
                y=y_pos,
                xerr=np.vstack([
                    top_skus["elasticity_mean"] - top_skus["ci_lower"],
                    top_skus["ci_upper"] - top_skus["elasticity_mean"]
                ]),
                fmt="o",
                capsize=5,
                markersize=8,
                markeredgewidth=1,
                markerfacecolor="white",
                markeredgecolor="black",
                ecolor="black",
                elinewidth=1,
                capthick=1
            )
            
            # Add SKU labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_skus["sku"])
            
            # Add vertical line at elasticity = -1
            ax.axvline(x=-1, color="red", linestyle="--", alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel("Price Elasticity")
            ax.set_title("Price Elasticity Estimates with 95% Credible Intervals")
            
            # Add grid
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            
            # Invert y-axis to have highest elasticities at the top
            ax.invert_yaxis()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            output_path = self.diagnostics_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved elasticity forest plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Forest plot creation failed: {str(e)}")
            
    def assess_convergence(self, trace: "az.InferenceData") -> bool:
        """
        Assess convergence of the MCMC chains.
        
        Args:
            trace: ArviZ InferenceData object with posterior samples
            
        Returns:
            True if chains have converged, False otherwise
        """
        # Compute diagnostics
        diagnostics = self.compute_diagnostics(trace)
        
        # Define convergence criteria
        converged = (
            diagnostics["rhat_max"] < 1.05 and  # Rhat below 1.05
            diagnostics["n_divergent"] == 0 and  # No divergences
            diagnostics["n_eff_min"] > 200  # Effective sample size sufficient
        )
        
        if converged:
            logger.info("MCMC chains have converged successfully")
        else:
            logger.warning("MCMC chains may not have converged properly")
            
        return converged 