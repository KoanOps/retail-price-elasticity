#!/usr/bin/env python3
"""
Tests for the decorators module.
"""
import unittest
import time
import logging
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to sys.path so we can import from utils
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.decorators import log_errors, log_step, timed


class TestDecorators(unittest.TestCase):
    """Tests for the decorators module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging for tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('test_logger')
        
    @patch('utils.decorators.logger')
    def test_log_errors_decorator(self, mock_logger):
        """Test the log_errors decorator."""
        
        @log_errors()
        def test_error_function():
            raise ValueError("Test error")
            
        # The function should raise the exception
        with self.assertRaises(ValueError):
            test_error_function()
            
        # Verify logger was called with error
        mock_logger.error.assert_called_once()
        
    @patch('utils.decorators.logger')
    def test_log_step_decorator(self, mock_logger):
        """Test the log_step decorator."""
        
        @log_step("Test Step")
        def test_step_function():
            return "step_result"
            
        result = test_step_function()
        
        # Verify logger was called for start and end
        self.assertEqual(mock_logger.info.call_count, 2)
        self.assertEqual(result, "step_result")
        
    @patch('utils.decorators.logger')
    def test_timed_decorator(self, mock_logger):
        """Test the timed decorator."""
        
        @timed()
        def test_timed_function():
            time.sleep(0.01)
            return "timed_result"
            
        result = test_timed_function()
        
        # Verify logger was called
        mock_logger.info.assert_called()
        self.assertEqual(result, "timed_result")
        
    @patch('utils.decorators.logger')
    def test_timed_decorator_with_params(self, mock_logger):
        """Test the timed decorator with parameters."""
        
        @timed("Custom Timing", log_level="debug")
        def test_timed_function_with_params(arg1, arg2=None):
            time.sleep(0.01)
            return f"{arg1}_{arg2}"
            
        result = test_timed_function_with_params("test", arg2="value")
        
        # Verify logger was called with debug level
        mock_logger.debug.assert_called()
        self.assertEqual(result, "test_value")


if __name__ == "__main__":
    unittest.main() 