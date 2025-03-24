"""
Python package configuration for retail-price-elasticity.

This setup script configures the package for distribution and installation,
defining metadata, dependencies, and entry points.
"""
from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Parse requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines()]

# Configure package metadata and dependencies
setup(
    # Basic package information
    name="retail-price-elasticity",  # Package name on PyPI
    version="0.1.0",                 # Semantic versioning
    author="Ryan Tong",              # Package author
    author_email="ryanyeetong@gmail.com",
    description="Comprehensive toolkit for analyzing price elasticity in retail data",
    
    # Detailed description for PyPI page
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Repository URL
    url="https://github.com/KoanOps/retail-price-elasticity",
    
    # Auto-detect all packages in the project
    packages=find_packages(),
    
    # PyPI classifiers for package categorization
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    
    # Python version requirements
    python_requires=">=3.8",
    
    # Dependencies from requirements.txt
    install_requires=requirements,
    
    # Command-line scripts that can be called after installation
    entry_points={
        "console_scripts": [
            "retail-analysis=main:main",  # Creates the 'retail-analysis' command
        ],
    },
) 