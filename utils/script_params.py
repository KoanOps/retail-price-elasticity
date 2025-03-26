"""Common parameter parsing functionality for scripts."""

import argparse
from typing import Callable, Optional

def get_base_parser() -> argparse.ArgumentParser:
    """Return base parser with common parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to data file')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--sample', type=float, help='Sample fraction')
    parser.add_argument('--draws', type=int, help='MCMC draws')
    parser.add_argument('--tune', type=int, help='MCMC tuning iterations')
    return parser

def parse_args_with_unknown(script_name: str, extra_args_func: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> argparse.Namespace:
    """Parse args and ignore unknown arguments.
    
    Args:
        script_name: Name of the script for help message
        extra_args_func: Optional function to add script-specific arguments
        
    Returns:
        Parsed arguments
    """
    base_parser = get_base_parser()
    if extra_args_func:
        extra_args_func(base_parser)
    args, _ = base_parser.parse_known_args()
    return args 