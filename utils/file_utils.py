#!/usr/bin/env python3
"""
File utility functions for the Retail package.

This module provides file and directory management utility functions.
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

from utils.logging_utils import logger


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensuring directory exists: {directory}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    ensure_dir_exists(directory)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved JSON data to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON file
    """
    if not os.path.exists(filepath):
        logger.warning(f"JSON file not found: {filepath}")
        return {}
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON data from {filepath}")
    return data


def get_file_extension(filepath: str) -> str:
    """
    Get file extension from filepath.
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (without dot)
    """
    return os.path.splitext(filepath)[1][1:].lower() 