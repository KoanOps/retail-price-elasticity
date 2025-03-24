#!/usr/bin/env python3
"""
Custom exceptions for the Retail Price Elasticity Analysis.

This module provides a hierarchy of exception classes tailored to various
error scenarios that may occur during model execution.
"""

class RetailError(Exception):
    """Base exception class for all retail analysis errors."""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# Data-related errors
class DataError(RetailError):
    """Error related to data loading, validation, or preparation."""
    pass


class DataFormatError(DataError):
    """Error related to data format or schema."""
    pass


class DataValidationError(DataError):
    """Error related to data validation."""
    pass


class DataPreparationError(DataError):
    """Error related to data preparation for modeling."""
    pass


# Model-related errors
class ModelError(RetailError):
    """Base class for model-related errors."""
    pass


class ModelBuildError(ModelError):
    """Error related to building a model."""
    pass


class SamplingError(ModelError):
    """Error related to MCMC sampling."""
    pass


class ModelEvaluationError(ModelError):
    """Error related to model evaluation."""
    pass


# Configuration-related errors
class ConfigurationError(RetailError):
    """Error related to configuration."""
    pass


# Execution-related errors
class ExecutionError(RetailError):
    """Error related to execution of the analysis."""
    pass


class RunnerError(ExecutionError):
    """Error related to the model runner."""
    pass


# Results-related errors
class ResultsError(RetailError):
    """Error related to results handling."""
    pass


# For backwards compatibility
VisualizationError = ModelError
FittingError = SamplingError  # Added for backward compatibility with bayesian_model.py 