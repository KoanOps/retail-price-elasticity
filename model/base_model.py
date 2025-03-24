#!/usr/bin/env python3
"""
Base model module for the SKU Elasticity Analysis.
Defines the abstract base class for all elasticity models.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

# Remove the manual sys.path insertion
from utils.logging_utils import logger, log_step
from config.default_config import get_config
from utils.common import ensure_dir_exists, save_json, load_json, to_serializable
from model.constants import (
    ELASTIC_THRESHOLD, 
    POSITIVE_ELASTICITY_THRESHOLD,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_DPI
)

# Use consolidated exceptions
from model.exceptions import (
    ModelError, DataError, ModelBuildError, 
    SamplingError, VisualizationError, ResultsError
)


class BaseElasticityModel(ABC):
    """
    Abstract base class for elasticity models.
    
    This class defines the interface that all elasticity models must implement.
    It provides common functionality such as:
    - Data preparation
    - Result storage and retrieval
    - Visualization utilities
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        model_name: str = "base_model",
        data_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base elasticity model.
        
        Parameters
        ----------
        results_dir : str
            Directory to save model results.
        model_name : str
            Name of the model for identification.
        data_config : dict, optional
            Configuration for data preparation.
        model_config : dict, optional
            Configuration for model parameters.
        """
        self.model_name = model_name
        self.data_config = data_config or {}
        self.model_config = model_config or {}
        
        # Set up results directories using pathlib.Path
        self.results_dir = Path(results_dir) / model_name
        self.model_dir = self.results_dir / 'model'
        self.diagnostics_dir = self.results_dir / 'diagnostics'
        
        # Create necessary directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model-specific attributes
        self.is_built = False
        self.model_data = None
        self.elasticity_estimates = None
        self.results = {}
        
        logger.info(f"Initialized {model_name} model in {results_dir}")
        
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> bool:
        """
        Prepare data for modeling.
        
        This method should:
        1. Validate input data
        2. Perform necessary preprocessing
        3. Handle missing values
        4. Create feature transformations
        5. Store the prepared data in self.model_data

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for modeling.

        Returns
        -------
        bool
            True if data preparation was successful, False otherwise.

        Raises
        ------
        DataError
            If there are issues with the input data.
        """
        pass
    
    @abstractmethod
    def build_model(self) -> bool:
        """
        Build the elasticity model.
        
        This method should:
        1. Set up the model structure
        2. Configure model parameters
        3. Prepare for estimation

        Returns
        -------
        bool
            True if model was built successfully, False otherwise.

        Raises
        ------
        ModelBuildError
            If there are issues building the model.
        """
        pass
    
    @abstractmethod
    def estimate_elasticities(self) -> Dict[str, Any]:
        """
        Estimate price elasticities.
        
        This method should:
        1. Execute the model
        2. Extract elasticity estimates
        3. Calculate confidence intervals
        4. Format results

        Returns
        -------
        dict
            Dictionary containing elasticity estimates and other results.

        Raises
        ------
        SamplingError
            If there are issues with sampling.
        ModelBuildError
            If the model has not been built.
        """
        pass
    
    def visualize_model(self) -> None:
        """
        Visualize the model structure.

        This method should:
        1. Create a visual representation of the model structure
        2. Save the visualization to the diagnostics directory

        Raises
        ------
        VisualizationError
            If there are issues generating visualizations.
        """
        raise NotImplementedError("Visualization not implemented for this model type")

    def generate_diagnostics(self) -> None:
        """
        Generate diagnostic visualizations and metrics.

        This method should:
        1. Create visualizations of the model posterior
        2. Evaluate model convergence and fit
        3. Save diagnostics to the diagnostics directory

        Raises
        ------
        VisualizationError
            If there are issues generating diagnostics.
        """
        raise NotImplementedError("Diagnostics not implemented for this model type")

    def summarize_results(self) -> Dict[str, Any]:
        """
        Summarize model results.

        Returns
        -------
        dict
            Dictionary with summary statistics and information.

        Raises
        ------
        ResultsError
            If there are issues generating the summary.
        """
        if self.elasticity_estimates is None:
            raise ResultsError("No elasticity estimates available")
        
        try:
            summary = {}
            
            # Extract elasticity values
            elasticities = []
            if isinstance(self.elasticity_estimates, dict) and 'elasticities' in self.elasticity_estimates:
                elasticity_dict = self.elasticity_estimates['elasticities']
                if isinstance(elasticity_dict, dict):
                    elasticities = list(elasticity_dict.values())
            
            if elasticities:
                # Calculate summary statistics
                summary["n_skus"] = len(elasticities)
                summary["mean_elasticity"] = float(np.mean(elasticities))
                summary["median_elasticity"] = float(np.median(elasticities))
                summary["min_elasticity"] = float(np.min(elasticities))
                summary["max_elasticity"] = float(np.max(elasticities))
                summary["std_elasticity"] = float(np.std(elasticities))
                
                # Count by elasticity ranges based on constants
                summary["elastic_count"] = sum(1 for e in elasticities if e < ELASTIC_THRESHOLD)
                summary["inelastic_count"] = sum(1 for e in elasticities if ELASTIC_THRESHOLD <= e < POSITIVE_ELASTICITY_THRESHOLD)
                summary["positive_count"] = sum(1 for e in elasticities if e >= POSITIVE_ELASTICITY_THRESHOLD)
                
                # Calculate percentages
                summary["elastic_percent"] = summary["elastic_count"] / summary["n_skus"] * 100
                summary["inelastic_percent"] = summary["inelastic_count"] / summary["n_skus"] * 100
                summary["positive_percent"] = summary["positive_count"] / summary["n_skus"] * 100
            else:
                summary["error"] = "No elasticity values found in results"
            
            return summary
            
        except TypeError as e:
            logger.error(f"Type error processing elasticity results: {str(e)}")
            raise ResultsError(f"Invalid elasticity data format: {str(e)}")
        except KeyError as e:
            logger.error(f"Missing key in elasticity results: {str(e)}")
            raise ResultsError(f"Missing required data in elasticity results: {str(e)}")
        except ValueError as e:
            logger.error(f"Value error processing elasticity results: {str(e)}")
            raise ResultsError(f"Invalid value in elasticity calculations: {str(e)}")
        except AttributeError as e:
            logger.error(f"Attribute error processing elasticity results: {str(e)}")
            raise ResultsError(f"Missing attribute in elasticity results: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error summarizing results: {str(e)}")
            raise ResultsError(f"Failed to summarize results: {str(e)}")

    def save_results(self, results: Dict[str, Any], filename: str = "model_results.json") -> str:
        """
        Save model results to file.

        Parameters
        ----------
        results : dict
            Results to save.
        filename : str
            Filename to save results to.

        Returns
        -------
        str
            Path to saved file.

        Raises
        ------
        ResultsError
            If there are issues saving the results.
        """
        try:
            # Create file path
            file_path = self.results_dir / filename
            
            # Convert results to serializable format
            serializable_results = to_serializable(results)
            
            # Save results
            save_json(serializable_results, file_path)
            logger.info(f"Saved model results to {file_path}")
            
            return str(file_path)
            
        except TypeError as e:
            logger.error(f"Type error saving results: {str(e)}")
            raise ResultsError(f"Results contain non-serializable type: {str(e)}")
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise ResultsError(f"Directory not found: {str(e)}")
        except PermissionError as e:
            logger.error(f"Permission error saving results: {str(e)}")
            raise ResultsError(f"No permission to write file: {str(e)}")
        except IOError as e:
            logger.error(f"I/O error saving results: {str(e)}")
            raise ResultsError(f"Failed to write results: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error saving results: {str(e)}")
            raise ResultsError(f"Failed to save results: {str(e)}")

    def _create_plot(self, 
                    data: Union[pd.DataFrame, np.ndarray], 
                    x_col: Optional[str] = None, 
                    y_col: Optional[str] = None,
                    plot_type: str = 'scatter',
                    title: str = '',
                    xlabel: str = '',
                    ylabel: str = '',
                    output_path: Optional[str] = None,
                    **kwargs) -> Optional[plt.Figure]:
        """
        Create a plot from the given data.
        
        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Data to plot.
        x_col : str, optional
            Name of the column to use for x-axis (if data is DataFrame).
        y_col : str, optional
            Name of the column to use for y-axis (if data is DataFrame).
        plot_type : str, default='scatter'
            Type of plot to create ('scatter' or 'line').
        title : str, default=''
            Title of the plot.
        xlabel : str, default=''
            Label for the x-axis.
        ylabel : str, default=''
            Label for the y-axis.
        output_path : str, optional
            Path to save the plot. If None, the plot will not be saved.
        **kwargs
            Additional keyword arguments to pass to the plot function.
            
        Returns
        -------
        matplotlib.figure.Figure or None
            Figure object if successful, None otherwise.
            
        Raises
        ------
        VisualizationError
            If there are issues creating the plot.
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                if x_col is None or y_col is None:
                    raise VisualizationError("For DataFrame input, x_col and y_col must be specified")
                
                # Plot based on type
                if plot_type.lower() == 'scatter':
                    ax.scatter(data[x_col], data[y_col], **kwargs)
                elif plot_type.lower() == 'line':
                    ax.plot(data[x_col], data[y_col], **kwargs)
                else:
                    raise VisualizationError(f"Unsupported plot type: {plot_type}")
            
            elif isinstance(data, np.ndarray):
                # For numpy arrays, check dimensions
                if len(data.shape) != 2 or data.shape[1] < 2:
                    raise VisualizationError("For array input, data must be 2D with at least 2 columns")
                
                # Plot based on type
                if plot_type.lower() == 'scatter':
                    ax.scatter(data[:, 0], data[:, 1], **kwargs)
                elif plot_type.lower() == 'line':
                    ax.plot(data[:, 0], data[:, 1], **kwargs)
                else:
                    raise VisualizationError(f"Unsupported plot type: {plot_type}")
            
            else:
                raise VisualizationError(f"Unsupported data type: {type(data)}")
            
            # Set plot attributes
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot if specified
            if output_path:
                plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
                logger.info(f"Plot saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            raise VisualizationError(f"Failed to create plot: {str(e)}")

    def _calculate_prediction_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     log_y_true: Optional[np.ndarray] = None,
                                     log_y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate prediction metrics.
        
        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
        log_y_true : array-like, optional
            True values in log space.
        log_y_pred : array-like, optional
            Predicted values in log space.
            
        Returns
        -------
        dict
            Dictionary of prediction metrics.
        """
        # Force numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics = {}
        
        # Original scale metrics
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        
        # Calculate MAPE, handling zero or near-zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_values = np.abs((y_true - y_pred) / y_true)
            # Replace infinities and NaNs with 1.0 (100% error)
            mape_values = np.where(np.isfinite(mape_values), mape_values, 1.0)
            metrics['mape'] = float(np.mean(mape_values) * 100)
        
        # R-squared
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        if ss_total > 0:
            metrics['r2'] = float(1 - (ss_residual / ss_total))
        else:
            metrics['r2'] = 0.0
        
        # Log space metrics if provided
        if log_y_true is not None and log_y_pred is not None:
            log_y_true = np.asarray(log_y_true)
            log_y_pred = np.asarray(log_y_pred)
            
            metrics['log_mse'] = float(np.mean((log_y_true - log_y_pred) ** 2))
            metrics['log_rmse'] = float(np.sqrt(metrics['log_mse']))
            metrics['log_mae'] = float(np.mean(np.abs(log_y_true - log_y_pred)))
            
            # Log space R-squared
            log_ss_total = np.sum((log_y_true - np.mean(log_y_true)) ** 2)
            log_ss_residual = np.sum((log_y_true - log_y_pred) ** 2)
            if log_ss_total > 0:
                metrics['log_r2'] = float(1 - (log_ss_residual / log_ss_total))
            else:
                metrics['log_r2'] = 0.0
        
        return metrics 
