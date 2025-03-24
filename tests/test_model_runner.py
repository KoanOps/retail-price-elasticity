#!/usr/bin/env python3
"""
Tests for the ModelRunner class.
"""
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from typing import Dict, Any, cast

from model.model_runner import ModelRunner
from model.exceptions import DataError, ModelBuildError, SamplingError


class TestModelRunner(unittest.TestCase):
    """Essential tests for ModelRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_path = "test_data.parquet"
        self.test_results_dir = "test_results"
        self.runner = ModelRunner(
            model_type="bayesian",
            data_path=self.test_data_path,
            results_dir=self.test_results_dir,
            visualize_only=False
        )

    def test_critical_error_handling(self):
        """Test critical error handling scenarios."""
        # Test data loading failure
        with patch('data.consolidated_data_loader.DataLoader') as mock_loader:
            mock_loader.side_effect = DataError("Invalid data path")
            with self.assertRaises(DataError):
                self.runner.run()

        # Test model creation failure
        self.runner.data = pd.DataFrame({'test': [1, 2, 3]})
        with patch('model.bayesian_model.BayesianModel') as mock_model:
            mock_model.side_effect = ModelBuildError("Invalid model config")
            with self.assertRaises(ModelBuildError):
                self.runner.run()

        # Test sampling failure
        self.runner.model = Mock()
        self.runner.model.estimate_elasticities.side_effect = SamplingError("Sampling failed")
        with self.assertRaises(SamplingError):
            self.runner.run()

    def test_type_safety(self):
        """Test type safety and null checks."""
        # Test initialization types
        self.assertIsNone(self.runner.data)
        self.assertIsNone(self.runner.model)
        self.assertIsInstance(self.runner.results, dict)

        # Skip the actual run since it would require data
        pass

    def test_pipeline_execution(self):
        """Test basic pipeline execution with mocks."""
        # Mock dependencies
        self.runner.data = pd.DataFrame({
            'Price_Per_Unit': [1.0, 2.0, 3.0],
            'Qty_Sold': [10, 20, 30],
            'SKU_Coded': ['A', 'B', 'C'],
            'Product_Class_Code': [1, 1, 2]
        })
        self.runner.model = Mock()
        self.runner.model.estimate_elasticities.return_value = {"status": "success"}

        # Test successful execution
        result = self.runner.run()
        self.assertIsInstance(result, dict)
        result_dict = cast(Dict[str, Any], result)
        self.assertEqual(result_dict["status"], "success")

        # Test visualization-only mode
        self.runner.visualize_only = True
        result = self.runner.run()
        self.assertIsInstance(result, dict)
        result_dict = cast(Dict[str, Any], result)
        self.assertEqual(result_dict["status"], "visualization_only")


if __name__ == "__main__":
    unittest.main() 