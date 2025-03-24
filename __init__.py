"""
Retail Price Elasticity Analysis Module

This package provides comprehensive tools for analyzing price elasticity
in retail data. The package includes components for:

- Data loading and preprocessing
- Model building and training
- Results analysis and visualization
- Utility functions for managing the analysis process

Main components:
- model: Contains the elasticity modeling capabilities
- data: Data loading and preprocessing tools
- config: Configuration management 
- utils: Utility functions for logging, timing, etc.
"""

__version__ = '0.1.0'
__author__ = 'Ryan Tong'

from model.model_runner import ModelRunner
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.data_visualizer import DataVisualizer
from config.config_manager import ConfigManager

__all__ = [
    'ModelRunner',
    'DataLoader',
    'DataPreprocessor',
    'DataVisualizer',
    'ConfigManager',
]
