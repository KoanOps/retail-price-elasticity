#!/usr/bin/env python3
"""
Results Manager for the Retail package.

This module provides functionality for saving, loading, and caching model results,
diagnostics, and visualizations.
"""

import os
import json
import pickle
from datetime import datetime
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import time

from utils.logging_utils import get_logger
from utils.decorators import log_errors
from utils.common import ensure_dir_exists, save_json, load_json, to_serializable
from utils.dependencies import get_arviz

# Get logger for this module
logger = get_logger()

class ResultsManager:
    """
    Manager for model results, diagnostics, and visualizations.
    
    Attributes:
        results_dir (str): Directory for storing results
        cache_dir (str): Directory for caching intermediate results
        metadata (Dict[str, Any]): Metadata about the results
    """
    
    def __init__(
        self,
        results_dir: str,
        experiment_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the ResultsManager.
        
        Args:
            results_dir: Base directory for storing results
            experiment_name: Name of the experiment (creates a subdirectory)
            cache_dir: Directory for caching intermediate results
            cache_enabled: Whether to enable caching
        """
        # Import arviz using dependency injection
        self.az = get_arviz()
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        # Set up directories
        self.results_dir = os.path.join(results_dir, experiment_name)
        ensure_dir_exists(self.results_dir)
        
        # Set up cache
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or os.path.join(self.results_dir, "cache")
        if cache_enabled:
            ensure_dir_exists(self.cache_dir)
        
        # Initialize metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results_files": {},
            "figures": {},
            "performance_metrics": {},
            "model_parameters": {},
            "data_stats": {}
        }
        
        logger.info(f"Initialized results manager for experiment '{experiment_name}'")
        logger.debug(f"Results directory: {self.results_dir}")
        if cache_enabled:
            logger.debug(f"Cache directory: {self.cache_dir}")
            
    @property
    def has_arviz(self) -> bool:
        """Check if ArviZ is available."""
        return self.az is not None
    
    @log_errors(msg="Error saving analysis results", reraise=True)
    def save_results(
        self, 
        results: Dict[str, Any], 
        results_dir: Optional[str] = None,
        result_type: str = "model", 
        filename_prefix: str = ""
    ) -> None:
        """
        Save analysis results of different types to the specified directory.
        
        This method consolidates the logic for saving different types of results
        (model results, elasticities, comparisons) to avoid code duplication.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results to save
        results_dir : Optional[str]
            Directory to save results in (defaults to self.results_dir)
        result_type : str
            Type of results: "model", "elasticity", "comparison"
        filename_prefix : str
            Optional prefix for output filenames
        """
        # Use instance results_dir if none provided
        if results_dir is None:
            results_dir = self.results_dir
            
        # Ensure the results directory exists
        results_path = Path(results_dir)
        ensure_dir_exists(results_path)
        
        # Add timestamp to filename if not already present
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        
        if result_type == "model":
            # Save model results
            
            # 1. Save trace if available
            if 'trace' in results and results['trace'] is not None:
                trace_dir = results_path / "traces"
                ensure_dir_exists(trace_dir)
                trace_file = trace_dir / f"{prefix}trace_{timestamp}.nc"
                
                try:
                    if self.has_arviz:
                        # Save as NetCDF file using arviz
                        self.az.to_netcdf(results['trace'], trace_file)
                        logger.info(f"Saved trace to {trace_file}")
                except Exception as e:
                    logger.error(f"Failed to save trace: {str(e)}")
            
            # 2. Save model summary if available
            if 'summary' in results:
                summary_file = results_path / f"{prefix}summary_{timestamp}.csv"
                try:
                    if isinstance(results['summary'], pd.DataFrame):
                        results['summary'].to_csv(summary_file)
                        logger.info(f"Saved summary to {summary_file}")
                    elif isinstance(results['summary'], dict):
                        pd.DataFrame.from_dict(results['summary']).to_csv(summary_file)
                        logger.info(f"Saved summary to {summary_file}")
                except Exception as e:
                    logger.error(f"Failed to save summary: {str(e)}")
            
            # 3. Save full results as JSON
            serializable_results = {
                k: v for k, v in results.items() 
                if k not in ['trace', 'model'] and v is not None
            }
            
            # Convert numpy/pandas objects to standard Python types
            serializable_results = to_serializable(serializable_results)
            
            # Save to JSON
            results_file = results_path / f"{prefix}results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            
            logger.info(f"Saved results to {results_file}")
            
        elif result_type == "elasticity":
            # Save elasticity results
            
            if 'elasticities' in results:
                elasticities = results['elasticities']
                
                # 1. Save elasticities as CSV
                elasticity_df = pd.DataFrame.from_dict(
                    elasticities, 
                    orient='index', 
                    columns=['elasticity']
                ).reset_index().rename(columns={'index': 'SKU'})
                
                elasticity_file = results_path / f"{prefix}elasticities_{timestamp}.csv"
                elasticity_df.to_csv(elasticity_file, index=False)
                
                # 2. Save summary statistics
                summary = {
                    'mean_elasticity': float(elasticity_df['elasticity'].mean()),
                    'median_elasticity': float(elasticity_df['elasticity'].median()),
                    'min_elasticity': float(elasticity_df['elasticity'].min()),
                    'max_elasticity': float(elasticity_df['elasticity'].max()),
                    'std_elasticity': float(elasticity_df['elasticity'].std()),
                    'sample_size': len(elasticity_df),
                    'elastic_count': sum(1 for e in elasticity_df['elasticity'] if e < -1.0),
                    'inelastic_count': sum(1 for e in elasticity_df['elasticity'] if -1.0 <= e < 0),
                    'positive_count': sum(1 for e in elasticity_df['elasticity'] if e >= 0)
                }
                
                # Add percentages
                summary['elastic_percent'] = summary['elastic_count'] / summary['sample_size'] * 100
                summary['inelastic_percent'] = summary['inelastic_count'] / summary['sample_size'] * 100
                summary['positive_percent'] = summary['positive_count'] / summary['sample_size'] * 100
                
                summary_file = results_path / f"{prefix}elasticity_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                    
                logger.info(f"Saved elasticities to {elasticity_file}")
                logger.info(f"Saved elasticity summary to {summary_file}")
            else:
                logger.warning("No elasticities found in results")
                
        elif result_type == "comparison":
            # Save comparison results
            
            if 'comparison' in results:
                comparison = results['comparison']
                
                # 1. Save as JSON
                comparison_file = results_path / f"{prefix}comparison_{timestamp}.json"
                with open(comparison_file, 'w') as f:
                    json.dump(to_serializable(comparison), f, indent=4)
                
                # 2. Save as CSV if it's tabular
                if isinstance(comparison, dict) and any(isinstance(v, dict) for v in comparison.values()):
                    try:
                        # Try to convert nested dict to DataFrame
                        comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
                        csv_file = results_path / f"{prefix}comparison_{timestamp}.csv"
                        comparison_df.to_csv(csv_file)
                        logger.info(f"Saved comparison CSV to {csv_file}")
                    except Exception as e:
                        logger.debug(f"Could not save comparison as CSV: {str(e)}")
                
                logger.info(f"Saved comparison to {comparison_file}")
            else:
                logger.warning("No comparison data found in results")
        else:
            logger.warning(f"Unknown result type: {result_type}")
    
    def save_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Save model parameters to metadata.
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.metadata["model_parameters"].update(parameters)
        logger.debug(f"Added {len(parameters)} model parameters to metadata")
    
    def save_data_stats(self, stats: Dict[str, Any]) -> None:
        """
        Save data statistics to metadata.
        
        Args:
            stats: Dictionary of data statistics
        """
        self.metadata["data_stats"].update(stats)
        logger.debug(f"Added {len(stats)} data statistics to metadata")
    
    def save_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save performance metrics to metadata.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.metadata["performance_metrics"].update(metrics)
        logger.debug(f"Added {len(metrics)} performance metrics to metadata")
    
    def save_dataframe(
        self, 
        df: pd.DataFrame, 
        name: str, 
        subdirectory: Optional[str] = None
    ) -> str:
        """
        Save a DataFrame to CSV and record in metadata.
        
        Args:
            df: DataFrame to save
            name: Name of the DataFrame
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Path to the saved file
        """
        # Set up path
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            ensure_dir_exists(directory)
            filename = os.path.join(directory, f"{name}.csv")
        else:
            filename = os.path.join(self.results_dir, f"{name}.csv")
        
        # Save the dataframe
        df.to_csv(filename, index=True)
        
        # Record in metadata
        rel_path = os.path.relpath(filename, self.results_dir)
        self.metadata["results_files"][name] = {
            "type": "dataframe",
            "path": rel_path,
            "rows": len(df),
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Saved DataFrame '{name}' to {filename}")
        return filename
    
    def save_figure(
        self, 
        figure: Figure, 
        name: str, 
        subdirectory: Optional[str] = None,
        formats: List[str] = ["png", "pdf"]
    ) -> List[str]:
        """
        Save a matplotlib figure in multiple formats.
        
        Args:
            figure: Matplotlib figure to save
            name: Base name for the figure
            subdirectory: Optional subdirectory within results_dir
            formats: List of formats to save (e.g., png, pdf, svg)
            
        Returns:
            List of paths to the saved files
        """
        # Set up directory
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            ensure_dir_exists(directory)
        else:
            directory = self.results_dir
        
        # Save in each format
        filenames = []
        for fmt in formats:
            filename = os.path.join(directory, f"{name}.{fmt}")
            figure.savefig(filename, bbox_inches="tight", dpi=300)
            filenames.append(filename)
        
        # Record in metadata
        self.metadata["figures"][name] = {
            "formats": formats,
            "paths": [os.path.relpath(f, self.results_dir) for f in filenames],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Saved figure '{name}' in formats: {formats}")
        return filenames
    
    def save_inference_data(
        self, 
        idata: Any,  # Type hint adjusted to avoid direct arviz reference
        name: str,
        subdirectory: Optional[str] = None
    ) -> str:
        """
        Save Arviz InferenceData object.
        
        Args:
            idata: Arviz InferenceData object
            name: Name for the dataset
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Path to the saved file
        """
        if not self.has_arviz:
            logger.warning("ArviZ not available, skipping InferenceData save")
            return ""
            
        # Set up path
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            ensure_dir_exists(directory)
            filename = os.path.join(directory, f"{name}.nc")
        else:
            filename = os.path.join(self.results_dir, f"{name}.nc")
        
        # Save the inference data
        idata.to_netcdf(filename)
        
        # Record in metadata
        rel_path = os.path.relpath(filename, self.results_dir)
        groups = list(idata._groups)
        self.metadata["results_files"][name] = {
            "type": "inference_data",
            "path": rel_path,
            "groups": groups,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Saved InferenceData '{name}' to {filename}")
        return filename
    
    def save_dict(
        self, 
        data: Dict[str, Any], 
        name: str,
        subdirectory: Optional[str] = None
    ) -> str:
        """
        Save a dictionary to JSON.
        
        Args:
            data: Dictionary to save
            name: Name for the dictionary
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Path to the saved file
        """
        # Set up path
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            ensure_dir_exists(directory)
            filename = os.path.join(directory, f"{name}.json")
        else:
            filename = os.path.join(self.results_dir, f"{name}.json")
        
        # Save the dictionary
        save_json(data, filename)
        
        # Record in metadata
        rel_path = os.path.relpath(filename, self.results_dir)
        self.metadata["results_files"][name] = {
            "type": "json",
            "path": rel_path,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Saved dictionary '{name}' to {filename}")
        return filename
    
    def save_object(
        self, 
        obj: Any, 
        name: str,
        subdirectory: Optional[str] = None
    ) -> str:
        """
        Save a Python object using pickle.
        
        Args:
            obj: Python object to save
            name: Name for the object
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Path to the saved file
        """
        # Set up path
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            ensure_dir_exists(directory)
            filename = os.path.join(directory, f"{name}.pkl")
        else:
            filename = os.path.join(self.results_dir, f"{name}.pkl")
        
        # Save the object
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        
        # Record in metadata
        rel_path = os.path.relpath(filename, self.results_dir)
        self.metadata["results_files"][name] = {
            "type": "pickle",
            "path": rel_path,
            "object_type": type(obj).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Saved object '{name}' to {filename}")
        return filename
    
    def save_metadata(self) -> str:
        """
        Save metadata to JSON file.
        
        Returns:
            Path to the saved file
        """
        filename = os.path.join(self.results_dir, "metadata.json")
        save_json(self.metadata, filename)
        logger.info(f"Saved experiment metadata to {filename}")
        return filename
    
    def load_dataframe(
        self, 
        name: str, 
        subdirectory: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a DataFrame from CSV.
        
        Args:
            name: Name of the DataFrame
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Loaded DataFrame
        """
        if subdirectory:
            filename = os.path.join(self.results_dir, subdirectory, f"{name}.csv")
        else:
            filename = os.path.join(self.results_dir, f"{name}.csv")
        
        if not os.path.exists(filename):
            logger.warning(f"DataFrame file not found: {filename}")
            return pd.DataFrame()
        
        df = pd.read_csv(filename, index_col=0)
        logger.info(f"Loaded DataFrame '{name}' from {filename}")
        return df
    
    def load_inference_data(
        self, 
        name: str,
        subdirectory: Optional[str] = None
    ) -> Any:  # Type hint adjusted to avoid direct arviz reference
        """
        Load Arviz InferenceData object.
        
        Args:
            name: Name of the dataset
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Arviz InferenceData object
        """
        if not self.has_arviz:
            logger.error("ArviZ not available, cannot load InferenceData")
            return None
            
        # Set up path
        if subdirectory:
            directory = os.path.join(self.results_dir, subdirectory)
            filename = os.path.join(directory, f"{name}.nc")
        else:
            filename = os.path.join(self.results_dir, f"{name}.nc")
        
        # Load the inference data
        if not os.path.exists(filename):
            logger.error(f"InferenceData file {filename} not found")
            return None
            
        try:
            idata = self.az.from_netcdf(filename)
            logger.info(f"Loaded InferenceData '{name}' from {filename}")
            return idata
        except Exception as e:
            logger.error(f"Error loading InferenceData: {str(e)}")
            return None
    
    def load_dict(
        self, 
        name: str,
        subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a dictionary from JSON.
        
        Args:
            name: Name of the dictionary
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Loaded dictionary
        """
        if subdirectory:
            filename = os.path.join(self.results_dir, subdirectory, f"{name}.json")
        else:
            filename = os.path.join(self.results_dir, f"{name}.json")
        
        return load_json(filename)
    
    def load_object(
        self, 
        name: str,
        subdirectory: Optional[str] = None
    ) -> Any:
        """
        Load a Python object using pickle.
        
        Args:
            name: Name of the object
            subdirectory: Optional subdirectory within results_dir
            
        Returns:
            Loaded object
        """
        if subdirectory:
            filename = os.path.join(self.results_dir, subdirectory, f"{name}.pkl")
        else:
            filename = os.path.join(self.results_dir, f"{name}.pkl")
        
        if not os.path.exists(filename):
            logger.warning(f"Object file not found: {filename}")
            return None
        
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"Loaded object '{name}' from {filename}")
        return obj
    
    def cached_compute(
        self, 
        compute_fn: Callable,
        cache_key: str,
        force_recompute: bool = False,
        **kwargs
    ) -> Any:
        """
        Compute a result with caching.
        
        Args:
            compute_fn: Function to compute the result
            cache_key: Key for caching the result
            force_recompute: Whether to force recomputation even if cached
            **kwargs: Arguments to pass to compute_fn
            
        Returns:
            Computed result
        """
        if not self.cache_enabled:
            return compute_fn(**kwargs)
        
        # Create a hash of the arguments
        args_str = json.dumps(to_serializable(kwargs), sort_keys=True)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()
        
        # Construct cache filename
        cache_filename = os.path.join(self.cache_dir, f"{cache_key}_{args_hash}.pkl")
        
        # Return cached result if available
        if not force_recompute and os.path.exists(cache_filename):
            logger.debug(f"Loading cached result for '{cache_key}'")
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)
        
        # Compute the result
        result = compute_fn(**kwargs)
        
        # Cache the result
        with open(cache_filename, 'wb') as f:
            pickle.dump(result, f)
        
        logger.debug(f"Cached result for '{cache_key}'")
        return result
    
    def list_results(self) -> Dict[str, List[str]]:
        """
        List available results by type.
        
        Returns:
            Dictionary mapping types to lists of result names
        """
        results = {}
        
        for name, info in self.metadata["results_files"].items():
            result_type = info.get("type", "unknown")
            if result_type not in results:
                results[result_type] = []
            results[result_type].append(name)
        
        return results


# For backward compatibility - using the class method with a default instance
def save_results(
    results: Dict[str, Any], 
    results_dir: str,
    result_type: str = "model", 
    filename_prefix: str = ""
) -> None:
    """
    Backward compatibility function for save_results.
    Creates a temporary ResultsManager instance and calls its save_results method.
    """
    manager = ResultsManager(results_dir=results_dir)
    manager.save_results(
        results=results,
        result_type=result_type,
        filename_prefix=filename_prefix
    )


# Testing code
if __name__ == "__main__":
    # Set up directories for testing
    test_dir = "test_results"
    ensure_dir_exists(test_dir)
    
    # Create results manager
    results_mgr = ResultsManager(results_dir=test_dir)
    
    # Test save/load methods
    test_dict = {"a": 1, "b": 2, "c": [1, 2, 3], "d": {"nested": "value"}}
    test_array = np.random.random((10, 3))
    
    # Save & load dictionary
    save_path = results_mgr.save_dict(test_dict, "test_dict")
    loaded_dict = results_mgr.load_dict("test_dict")
    assert loaded_dict["a"] == test_dict["a"]
    
    # Save & load object
    obj_path = results_mgr.save_object(test_array, "test_array")
    loaded_array = results_mgr.load_object("test_array")
    assert np.array_equal(loaded_array, test_array)
    
    # Test figure saving
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    fig_path = results_mgr.save_figure(fig, "test_figure", subdirectory="plots")
    assert os.path.exists(fig_path)
    
    # Test caching
    def expensive_computation(x):
        return x ** 2
    
    cached_result = results_mgr.cached_compute(
        expensive_computation,
        "square_calculation",
        x=42
    )
    assert cached_result == 42 ** 2
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("Tests completed successfully!") 