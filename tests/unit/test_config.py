import pytest
from unittest.mock import patch, mock_open
import json
import os

import config

class TestConfigLoading:
    @patch('os.path.dirname')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_config_success(self, mock_file, mock_dirname, mock_config):
        """Test successful config loading from file"""
        # Setup mocks
        mock_dirname.return_value = '/test/path'
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
        
        # Call the function
        result = config.loadConfig()
        
        # Verify correct config was loaded
        assert result == mock_config
        assert 'mode' in result
        assert 'discord' in result
        assert 'default_stock' in result
        
    @patch('os.path.dirname')
    @patch('builtins.open')
    def test_load_config_file_not_found(self, mock_file, mock_dirname):
        """Test fallback to default config when file not found"""
        # Setup mock to raise FileNotFoundError
        mock_dirname.return_value = '/test/path'
        mock_file.side_effect = FileNotFoundError()
        
        # Call the function
        result = config.loadConfig()
        
        # Verify default config is returned
        assert isinstance(result, dict)
        assert 'mode' in result
        assert 'discord' in result
        assert 'default_stock' in result
        
    @patch('os.path.dirname')
    @patch('builtins.open')
    def test_load_config_invalid_json(self, mock_file, mock_dirname):
        """Test fallback to default config when JSON is invalid"""
        # Setup mock to return invalid JSON
        mock_dirname.return_value = '/test/path'
        mock_file_handle = mock_open(read_data='{invalid json')()
        mock_file.return_value = mock_file_handle
        
        # Call the function
        result = config.loadConfig()
        
        # Verify default config is returned
        assert isinstance(result, dict)
        assert 'mode' in result
        assert 'discord' in result
        assert 'default_stock' in result

class TestArgumentValidation:
    def test_valid_arguments(self, mock_config):
        """Test validation with valid arguments"""
        # Valid period and interval
        period = "1y"
        interval = "1d"
        
        valid, error_msg = config.validateArgs(period, interval, mock_config)
        
        assert valid is True
        assert error_msg is None
        
    def test_invalid_period(self, mock_config):
        """Test validation with invalid period"""
        # Invalid period
        period = "2d"  # Not in allowed periods
        interval = "1d"
        
        valid, error_msg = config.validateArgs(period, interval, mock_config)
        
        assert valid is False
        assert "Nieprawidłowy okres" in error_msg
        
    def test_invalid_interval(self, mock_config):
        """Test validation with invalid interval"""
        # Invalid interval
        period = "1y"
        interval = "3d"  # Not in allowed intervals
        
        valid, error_msg = config.validateArgs(period, interval, mock_config)
        
        assert valid is False
        assert "Nieprawidłowy interwał" in error_msg
        
    def test_incompatible_period_interval(self, mock_config):
        """Test validation with incompatible period-interval combination"""
        # Long period with short interval
        period = "1y"  # Long period
        interval = "1m"  # Short interval
        
        valid, error_msg = config.validateArgs(period, interval, mock_config)
        assert valid is False
        assert "Nieprawidłowy" in error_msg
        assert "Dozwolone" in error_msg
