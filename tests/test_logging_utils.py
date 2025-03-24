#!/usr/bin/env python3
"""
Tests for the logging_utils module.
"""
import unittest
import logging
import json
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to sys.path so we can import from utils
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.logging_utils import LoggingManager, logger, log_step, log_function


class TestLoggingUtils(unittest.TestCase):
    """Tests for the logging_utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset logger for each test
        logger.handlers = []
        logger.setLevel(logging.INFO)
        
    def test_setup_logging(self):
        """Test the setup_logging function."""
        test_logger = LoggingManager.setup_logging(
            logger_name="test_logger",
            log_level=logging.DEBUG,
            log_format="%(message)s",
            log_file=None
        )
        
        self.assertEqual(test_logger.name, "test_logger")
        self.assertEqual(test_logger.level, logging.DEBUG)
        self.assertEqual(len(test_logger.handlers), 1)  # Console handler
        
    def test_log_dict(self):
        """Test log_dict method."""
        # Patch the logger to capture the log message
        with patch.object(logger, 'info') as mock_info:
            # Call log_dict
            test_data = {"a": 1, "b": 2}
            LoggingManager.log_dict(logger, "Test Dict", test_data)
            
            # Verify the logger was called with formatted JSON
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn("Test Dict", call_args)
            self.assertIn(json.dumps(test_data, indent=2), call_args)
            
    @patch('utils.logging_utils.logger')
    def test_log_step_decorator(self, mock_logger):
        """Test the log_step decorator."""
        
        @log_step("Running test function")
        def test_function():
            return "test_result"
            
        result = test_function()
        
        # Verify the logger was called
        self.assertEqual(mock_logger.info.call_count, 2)  # Start and end messages
        self.assertEqual(result, "test_result")

    @patch('utils.logging_utils.logger')
    def test_log_function_decorator(self, mock_logger):
        """Test the log_function decorator."""
        
        @log_function()
        def test_function(arg1, arg2=None):
            return f"{arg1}_{arg2}"
            
        result = test_function("test", arg2="value")
        
        # Verify the logger was called
        self.assertEqual(mock_logger.info.call_count, 2)  # Start and end messages
        self.assertEqual(result, "test_value")


if __name__ == "__main__":
    unittest.main() 