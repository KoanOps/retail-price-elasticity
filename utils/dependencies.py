"""
Centralized dependency management for the retail elasticity analysis.

This module handles importing optional dependencies using dependency injection
rather than global state.

Note: Dependency checking is not done automatically on import.
To check dependencies, call dependency_manager.check_dependencies()
in your initialization code.
"""

import logging
import importlib
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache

from utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger()

class DependencyManager:
    """
    Manages optional dependencies for the application.
    
    This class provides methods to check for dependencies and import them
    on demand, using dependency injection rather than global state.
    """
    
    def __init__(self):
        """Initialize the dependency manager."""
        self.dependency_status = {}
        self.modules = {}
        
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check all optional dependencies and log their status.
        
        Returns:
            Dictionary mapping dependency names to availability status.
        """
        dependencies = [
            ("pymc", "PyMC", "Bayesian modeling will be disabled."),
            ("arviz", "ArviZ", "Bayesian model diagnostics will be limited."),
            ("statsmodels", "statsmodels", "Some statistical tests will be disabled.")
        ]
        
        for module_name, display_name, message in dependencies:
            self._check_single_dependency(module_name, display_name, message)
            
        # Check visualization dependencies separately
        self.check_visualization_libs()
        
        # Log availability summary
        logger.info("Dependency status:")
        for dep, available in self.dependency_status.items():
            status = "Available" if available else "Not available"
            logger.info(f"  {dep}: {status}")
            
        return self.dependency_status.copy()
        
    def check_visualization_libs(self) -> bool:
        """
        Check if visualization libraries are available.
        
        Returns:
            True if all required visualization libraries are available, False otherwise.
        """
        # Check for matplotlib
        matplotlib_available = self._check_single_dependency("matplotlib.pyplot", "Matplotlib",
                                                "Basic plotting will be disabled.")
        
        # Check for seaborn (depends on matplotlib)
        seaborn_available = False
        if matplotlib_available:
            seaborn_available = self._check_single_dependency("seaborn", "Seaborn",
                                                 "Advanced plotting will be limited.")
        
        # Consider visualization available if at least matplotlib is available
        self.dependency_status["visualization"] = matplotlib_available
        
        return matplotlib_available
        
    def _check_single_dependency(self, module_name: str, display_name: str, missing_message: str) -> bool:
        """
        Check for a single dependency and log appropriate messages.
        
        Args:
            module_name: Name of the module to import
            display_name: Display name for logging
            missing_message: Message to log if dependency is missing
            
        Returns:
            True if dependency is available, False otherwise
        """
        try:
            if module_name not in self.modules:
                self.modules[module_name] = importlib.import_module(module_name)
            self.dependency_status[display_name] = True
            return True
        except ImportError:
            logger.warning(f"{display_name} not available. {missing_message}")
            self.dependency_status[display_name] = False
            return False
            
    def get_module(self, module_name: str) -> Optional[Any]:
        """
        Get a module by name if it's available.
        
        Args:
            module_name: Name of the module to get
            
        Returns:
            Module object if available, None otherwise
        """
        if module_name in self.modules:
            return self.modules[module_name]
            
        try:
            module = importlib.import_module(module_name)
            self.modules[module_name] = module
            return module
        except ImportError:
            return None

    def check_pymc(self) -> bool:
        """
        Check if PyMC is available.
        
        Returns:
            True if PyMC is available, False otherwise
        """
        return self._check_single_dependency("pymc", "PyMC", 
                                           "Bayesian modeling will be disabled.")
                                           
    def check_arviz(self) -> bool:
        """
        Check if ArviZ is available.
        
        Returns:
            True if ArviZ is available, False otherwise
        """
        return self._check_single_dependency("arviz", "ArviZ", 
                                           "Bayesian model diagnostics will be limited.")


# Create a singleton instance for the application
@lru_cache(maxsize=1)
def get_dependency_manager() -> DependencyManager:
    """Get the singleton instance of the dependency manager."""
    return DependencyManager()


# Public API functions
def check_dependencies() -> Dict[str, bool]:
    """Check all dependencies."""
    return get_dependency_manager().check_dependencies()

def get_pymc():
    """Get PyMC module if available."""
    return get_dependency_manager().get_module("pymc")

def get_arviz():
    """Get ArviZ module if available."""
    return get_dependency_manager().get_module("arviz") 