#!/usr/bin/env python3
"""
Serialization utilities for the Retail package.

This module provides functions for serializing and deserializing Python objects.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

from utils.logging_utils import logger


def to_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable representation of object
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return {
            'columns': list(obj.columns),
            'index': list(obj.index),
            'data': obj.values.tolist()
        }
    elif isinstance(obj, pd.Series):
        return {
            'name': obj.name,
            'index': list(obj.index),
            'data': obj.values.tolist()
        }
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Try to convert to string for objects that don't fit other categories
        try:
            return str(obj)
        except Exception:
            logger.warning(f"Cannot serialize object of type {type(obj)}")
            return None 