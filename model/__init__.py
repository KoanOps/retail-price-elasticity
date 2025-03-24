"""
Model package for Retail Price Elasticity Analysis.

This package provides model implementation, estimation, and evaluation
functionality for retail price elasticity analysis.
"""

# Import only essential base classes to avoid circular imports
from model.base_model import BaseElasticityModel
from model.exceptions import ModelError, DataError, ModelBuildError

# Import model components directly
from model.data_preparation import BayesianDataPreparation
from model.bayesian.model_builder import BayesianModelBuilder
from model.sampling import BayesianSampler
from model.diagnostics import BayesianDiagnostics
from model.visualization import BayesianVisualizer

# Re-export main classes for easier import by consumers
__all__ = [
    'ModelRunner', 'BaseElasticityModel',
    'ModelError', 'DataError', 'ModelBuildError',
    'BayesianModelBuilder', 'BayesianDataPreparation',
    'BayesianSampler', 'BayesianDiagnostics', 'BayesianVisualizer',
]

# Import model_runner after defining BaseElasticityModel to avoid circular imports
from model.model_runner import ModelRunner 