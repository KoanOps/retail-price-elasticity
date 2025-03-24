#!/usr/bin/env python3
"""
Common utility functions for the Retail package.

This module provides common utility functions used throughout the codebase.
Functions here should be general-purpose and reusable across modules.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, date

from utils.logging_utils import logger
from utils.file_utils import ensure_dir_exists, save_json, load_json, get_file_extension
from utils.data_utils import (
    DataTransformer, validate_column_exists, validate_columns_exist,
    calculate_summary_stats, split_train_test, batch_process
)
from utils.serialization import to_serializable

# Export all imported functions to maintain API compatibility
__all__ = [
    'ensure_dir_exists', 'save_json', 'load_json', 'get_file_extension',
    'DataTransformer', 'validate_column_exists', 'validate_columns_exist',
    'calculate_summary_stats', 'split_train_test', 'batch_process',
    'to_serializable'
]

# Add any remaining utility functions that don't fit into the other modules

def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format a number with specified decimal places and commas for thousands.
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.{decimal_places}f}"
    return str(value)

def to_serializable(obj):
    """Convert an object to a JSON serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, 'asdict'):
        return obj.asdict()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    else:
        return obj
        
def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key of the parent dictionary
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Add simple tests
if __name__ == "__main__":
    # Test directory creation
    test_dir = "test_dir"
    ensure_dir_exists(test_dir)
    assert os.path.exists(test_dir)
    
    # Test JSON serialization
    test_data = {
        'int_val': 42,
        'float_val': 3.14,
        'array_val': np.array([1, 2, 3]),
        'nested': {
            'a': 1,
            'b': 2
        }
    }
    
    serialized = to_serializable(test_data)
    assert isinstance(serialized['array_val'], list)
    
    # Test number formatting
    assert format_number(1234.56789) == "1,234.57"
    assert format_number(1234) == "1,234"
    
    # Test file extension
    assert get_file_extension("/path/to/file.txt") == "txt"
    assert get_file_extension("file.JSON") == "json"
    
    # Clean up
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
    
    logger.info("All tests passed!") 