import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture
def sample_stock_data():
    """Load raw stock data from the test dummy file"""
    # Load from the existing test data file
    file_path = os.path.join(os.path.dirname(__file__), 'AAPL_1y_1d_raw.csv')
    return pd.read_csv(file_path)

@pytest.fixture
def processed_stock_data():
    """Load processed stock dataframe with features from test dummy file"""
    # Load from the existing processed test data file
    file_path = os.path.join(os.path.dirname(__file__), 'AAPL_1y_1d_processed.csv')
    return pd.read_csv(file_path)

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return {
        "mode": "discord",
        "discord": {
            "token": "test_token",
            "command_prefix": "!"
        },
        "default_stock": {
            "ticker": "AAPL",
            "period": "1y",
            "interval": "1d"
        },
        "news_sentiment": {
            "enabled": True,
            "storiesCount": 5,
            "daysLookback": 7,
            "sentiment_impact_weight": 0.5
        },
        "allowed_arguments": {
            "intervals": ["1d", "5d", "1wk", "1mo"],
            "periods": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
            "long_periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            "short_intervals": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        }
    }

@pytest.fixture
def sample_news_stories():
    """Create sample news stories for testing"""
    from datetime import datetime
    
    return [
        {
            'id': '1',
            'time': datetime(2023, 1, 15, 10, 30),
            'url': 'https://example.com/news/1',
            'site': 'Example Finance',
            'title': 'AAPL Reports Strong Quarterly Earnings',
            'description': 'Apple Inc. reported better than expected earnings for Q4 2022.'
        },
        {
            'id': '2',
            'time': datetime(2023, 1, 14, 14, 15),
            'url': 'https://example.com/news/2',
            'site': 'Market News',
            'title': 'New iPhone Model Announcement Expected',
            'description': 'Apple is rumored to announce a new iPhone model next month.'
        },
        {
            'id': '3',
            'time': datetime(2023, 1, 13, 9, 45),
            'url': 'https://example.com/news/3',
            'site': 'Tech Today',
            'title': 'AAPL Stock Dips on Supply Chain Concerns',
            'description': 'Apple shares fell 2% following reports of supply chain disruptions.'
        }
    ]
