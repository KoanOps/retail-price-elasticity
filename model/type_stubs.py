"""
Type stubs for PyMC and ArviZ objects.

This module provides type hints and runtime type checking for PyMC and ArviZ objects.
"""
from typing import Dict, Any, Optional, Protocol, runtime_checkable, TypeVar, Union, cast
from typing_extensions import TypeAlias

# Type aliases for common patterns
ModelData: TypeAlias = Dict[str, Any]
PosteriorDict: TypeAlias = Dict[str, Any]
ModelVars: TypeAlias = Dict[str, Any]
TraceData: TypeAlias = Dict[str, Any]

# Generic type variables
T = TypeVar('T')

# Type for PyMC's treedict
Treedict = Any  # We use Any here since treedict is a PyMC internal type

@runtime_checkable
class Model(Protocol):
    """Protocol for PyMC Model objects."""
    
    named_vars: Union[Dict[str, Any], Treedict]  # Allow both dict and treedict
    
    def __enter__(self) -> 'Model':
        """Context manager entry."""
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...

@runtime_checkable
class Posterior(Protocol):
    """Protocol for ArviZ Posterior objects."""
    
    values: Any
    
    def mean(self, axis: Optional[tuple[int, ...]] = None) -> Any:
        """Calculate mean along specified axes."""
        ...
    
    def std(self, axis: Optional[tuple[int, ...]] = None) -> Any:
        """Calculate standard deviation along specified axes."""
        ...

# Type alias for posterior-like objects
PosteriorLike: TypeAlias = Union[Posterior, Any]  # Allow both Posterior and raw values

@runtime_checkable
class InferenceData(Protocol):
    """Protocol for ArviZ InferenceData objects."""
    
    posterior: Posterior
    
    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        ...

# Type checking utilities
def is_valid_model(model: Optional[Model]) -> bool:
    """Check if model is valid and has required attributes."""
    if model is None:
        return False
    return hasattr(model, 'named_vars') and hasattr(model, '__enter__') and hasattr(model, '__exit__')

def is_valid_inference_data(inference_data: Optional[InferenceData]) -> bool:
    """Check if inference data is valid and has required attributes."""
    if inference_data is None:
        return False
    return hasattr(inference_data, 'posterior') and hasattr(inference_data, '__getitem__')

def is_valid_posterior(posterior: Optional[PosteriorLike]) -> bool:
    """Check if posterior is valid and has required attributes."""
    if posterior is None:
        return False
    return hasattr(posterior, 'values') and hasattr(posterior, 'mean') and hasattr(posterior, 'std')

def is_valid_model_data(model_data: Optional[ModelData]) -> bool:
    """Check if model data is valid and has required keys."""
    if model_data is None:
        return False
    required_keys = {'sku_idx', 'log_price', 'log_qty'}
    return all(key in model_data for key in required_keys)

# Type casting utilities
def cast_to_inference_data(obj: Any) -> Optional[InferenceData]:
    """Cast object to InferenceData if valid."""
    if is_valid_inference_data(obj):
        return cast(InferenceData, obj)
    return None

def cast_to_posterior(obj: Any) -> Optional[Posterior]:
    """Cast object to Posterior if valid."""
    if is_valid_posterior(obj):
        return cast(Posterior, obj)
    return None

def cast_to_model_data(obj: Any) -> Optional[ModelData]:
    """Cast object to ModelData if valid."""
    if is_valid_model_data(obj):
        return cast(ModelData, obj)
    return None

# Safe attribute access utilities
def get_model_vars(model: Optional[Model]) -> Dict[str, Any]:
    """Safely get model variables."""
    if not is_valid_model(model):
        return {}
    return getattr(model, 'named_vars', {})

def get_posterior_dict(inference_data: Optional[InferenceData]) -> Dict[str, Any]:
    """Safely get posterior dictionary."""
    if not is_valid_inference_data(inference_data):
        return {}
    return getattr(inference_data, 'posterior', {})

def get_model_data_shape(model_data: Optional[ModelData]) -> Optional[tuple[int, ...]]:
    """Safely get model data shape."""
    if not is_valid_model_data(model_data):
        return None
    try:
        if model_data is None:
            return None
        return tuple(len(model_data[key]) for key in ['sku_idx', 'log_price', 'log_qty'])
    except (KeyError, TypeError):
        return None

def get_model_data_row(model_data: Optional[ModelData], idx: int) -> Optional[Dict[str, Any]]:
    """Safely get a row from model data."""
    if not is_valid_model_data(model_data):
        return None
    try:
        if model_data is None:
            return None
        return {key: model_data[key][idx] for key in ['sku_idx', 'log_price', 'log_qty']}
    except (KeyError, IndexError):
        return None 