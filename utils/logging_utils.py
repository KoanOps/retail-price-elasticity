#!/usr/bin/env python3
"""
Logging utilities for the Retail Elasticity Analysis.

This module provides:
1. Consistent logging setup across the application
2. Decorators for structured logging
3. Helper methods for common logging patterns
"""
import logging
import os
import json
import sys
import functools
import datetime
import traceback
import time
from typing import Dict, Any, Optional, Callable, TypeVar, cast
from pathlib import Path

# Type variables for callable
F = TypeVar('F', bound=Callable[..., Any])

class LoggerProvider:
    """
    Provides centralized access to the application logger.
    
    This class ensures that all components use the same logger instance
    and avoids global state by using a singleton pattern.
    """
    _instance = None
    _logger = None
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Get the application logger instance.
        
        Returns:
            The application logger
        """
        if cls._logger is None:
            # Create a default logger if not yet configured
            cls._logger = logging.getLogger('Retail_Elasticity')
            if not cls._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
                cls._logger.setLevel(logging.INFO)
        
        return cls._logger

# For backward compatibility and ease of use
def get_logger() -> logging.Logger:
    """Get the application logger."""
    return LoggerProvider.get_logger()

# Initialize logger for module-level functions to use
logger = get_logger()

def log_step(step_name: str = None) -> Callable[[F], F]:
    """
    Decorator to log the start and end of a step with timing information.
    
    Can be used with or without a step name:
    
    @log_step
    def my_func():
        ...
        
    @log_step("Processing Data")
    def my_func():
        ...

    Args:
        step_name: Optional name of the processing step. If None, function name is used.

    Returns:
        Decorated function that logs step start and end
    """
    def decorator(func: F) -> F:
        nonlocal step_name
        if step_name is None:
            step_name = func.__name__
            
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get logger
            log = get_logger()
            
            # Log start
            log.info(f"Starting step: {step_name}")
            start_time = time.time()
            
            # Call function
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                log.error(f"Error in step {step_name}: {str(e)}")
                raise
            finally:
                # Log end with timing
                elapsed = time.time() - start_time
                status = "completed successfully" if success else "failed"
                log.info(f"Step {step_name} {status} in {elapsed:.2f} seconds")
                
            return result
            
        return wrapper
    
    # Handle case where decorator is used without arguments: @log_step
    if callable(step_name):
        func, step_name = step_name, step_name.__name__
        return decorator(func)
    
    # Handle case with arguments: @log_step("step name")
    return decorator

class LoggingManager:
    """
    Manages logging configuration and provides utility methods for logging.
    
    This class offers methods to:
    - Set up logging with consistent formatting
    - Create standardized log messages for steps, warnings, and errors
    - Handle suppression of specific warnings
    """
    
    @staticmethod
    def setup_logging(
        logger_name: str = 'Retail_Elasticity',
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        suppress_warnings: bool = False,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            logger_name: Name of the logger
            log_level: Logging level (default: INFO)
            log_file: Path to log file (if None, logs to console only)
            suppress_warnings: Whether to suppress python warnings
            log_format: Format string for log messages
            
        Returns:
            Configured logger instance
        """
        # Get the logger
        log = logging.getLogger(logger_name)
        log.setLevel(log_level)
        
        # Remove existing handlers
        if log.handlers:
            log.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        
        # Create file handler if log_file is specified
        if log_file:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            # Create the file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)
        
        # Suppress warnings if requested
        if suppress_warnings:
            import warnings
            warnings.filterwarnings('ignore')
        
        # Update the main logger in LoggerProvider
        LoggerProvider._logger = log
        
        return log
    
    @staticmethod
    def log_step_start(log: logging.Logger, step_name: str) -> None:
        """
        Log the start of a processing step with a standardized format.
        
        Args:
            log: Logger instance
            step_name: Name of the processing step
        """
        log.info(f"\n{'-'*20} STARTING: {step_name} {'-'*20}")
    
    @staticmethod
    def log_step_end(log: logging.Logger, step_name: str) -> None:
        """
        Log the end of a processing step with a standardized format.
        
        Args:
            log: Logger instance
            step_name: Name of the processing step
        """
        log.info(f"{'-'*20} COMPLETED: {step_name} {'-'*20}\n")
    
    @staticmethod
    def log_warning(
        log: logging.Logger, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a warning with optional data.
        
        Args:
            log: Logger instance
            message: Warning message
            data: Optional dictionary with additional data
        """
        if data:
            log.warning(f"{message} | Data: {data}")
        else:
            log.warning(message)
    
    @staticmethod
    def log_error(
        log: logging.Logger, 
        message: str, 
        exception: Optional[Exception] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error with optional exception and data.
        
        Args:
            log: Logger instance
            message: Error message
            exception: Optional exception object
            data: Optional dictionary with additional data
        """
        if exception:
            log.error(f"{message}: {str(exception)}")
            if hasattr(exception, '__traceback__'):
                import traceback
                log.debug(''.join(traceback.format_tb(exception.__traceback__)))
        else:
            log.error(message)
            
        if data:
            log.error(f"Error data: {data}")
    
    @staticmethod
    def log_dataframe_info(
        log: logging.Logger,
        df_name: str,
        df: Any,  # Using Any instead of pd.DataFrame to avoid the linter error
        include_stats: bool = False
    ) -> None:
        """
        Log information about a pandas DataFrame.
        
        Args:
            log: Logger instance
            df_name: Name of the DataFrame
            df: The pandas DataFrame
            include_stats: Whether to include basic statistics
        """
        log.info(f"DataFrame '{df_name}' shape: {df.shape}")
        log.info(f"DataFrame '{df_name}' columns: {list(df.columns)}")
        
        # Log NA values
        na_counts = df.isna().sum()
        if na_counts.sum() > 0:
            log.info(f"DataFrame '{df_name}' NA counts:\n{na_counts[na_counts > 0]}")
        
        # Log basic statistics if requested
        if include_stats:
            try:
                log.info(f"DataFrame '{df_name}' statistics:\n{df.describe().to_string()}")
            except Exception as e:
                log.warning(f"Could not generate statistics for DataFrame '{df_name}': {str(e)}")
                
    @staticmethod
    def log_dict(
        log: logging.Logger,
        title: str,
        data: Dict[str, Any],
        level: str = 'info'
    ) -> None:
        """
        Log a dictionary with a title.
        
        Args:
            log: Logger instance
            title: Title for the log entry
            data: Dictionary to log
            level: Log level ('debug', 'info', 'warning', 'error')
        """
        log_method = getattr(log, level.lower())
        
        # Format the dictionary
        formatted_data = "\n".join([f"  {k}: {v}" for k, v in data.items()])
        
        # Log the data
        log_method(f"{title}:\n{formatted_data}") 