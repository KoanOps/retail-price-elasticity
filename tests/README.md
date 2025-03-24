# Test Suite for Retail Price Elasticity Analysis

This directory contains comprehensive test and validation tools for the Retail Price Elasticity Analysis project. These tests ensure the reliability, accuracy, and performance of the elasticity modeling components.

## Test Categories

The test suite is organized into three main categories:

### 1. Unit Tests

These tests verify the correctness of individual components and functions:

- `test_model_runner.py` - Tests for the ModelRunner orchestration framework
- `test_logging_utils.py` - Tests for the logging infrastructure
- `test_decorators.py` - Tests for function decorators (timing, error handling, etc.)

### 2. Validation Tools

These tools validate model accuracy against known ground truth:

- `validate_elasticity_model.py` - Validates elasticity models against simulated data with known elasticities
  
### 3. Integration Tests

These tests verify that system components work together correctly:

- Testing data loading with model execution
- End-to-end processing pipelines

## Running the Tests

### Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all unit tests
pytest tests/test_*.py

# Run tests with coverage
pytest tests/test_*.py --cov=.
```

### Model Validation

```bash
# Run model validation with simulated data
python tests/validate_elasticity_model.py \
  --model-type bayesian \
  --observations 10000 \
  --skus 50 \
  --sample-frac 0.3 \
  --draws 1000 \
  --tune 500 \
  --results-dir results/validation_test
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. **For unit tests:**
   - Create a file named `test_<module>.py`
   - Use pytest fixtures for common setup
   - Group related tests into classes
   - Include both positive tests and edge cases

2. **For validation tests:**
   - Create descriptive, standalone scripts
   - Include command-line options for configuration
   - Save results to a specified output directory
   - Provide clear reporting of metrics

## Continuous Integration

All unit tests are automatically run on pull requests and commits to the main branch as part of our CI/CD pipeline. Validation tests should be run manually when making significant changes to model components.

## Performance Benchmarks

The validation tools can also serve as performance benchmarks. When changing model implementations, compare validation metrics before and after your changes to ensure continued accuracy. 