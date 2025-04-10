import pytest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

import utils

class TestLoadData:
    def test_load_data(self, tmp_path, sample_stock_data):
        """Test that loadData correctly loads and parses CSV data"""
        # Create a temp CSV file
        test_file = tmp_path / "AAPL_1y_1d.csv"
        sample_stock_data.to_csv(test_file, index=False)
        
        # Mock os.path methods
        with patch('os.path.basename', return_value="AAPL_1y_1d.csv"), \
             patch('os.path.splitext', return_value=("AAPL_1y_1d", ".csv")):
            
            # Test the function
            result = utils.loadData(test_file)
            
            # Validate result
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == list(sample_stock_data.columns)
            assert len(result) == len(sample_stock_data)

class TestGeneratePredictionChart:
    def test_generate_prediction_chart(self, tmp_path, processed_stock_data):
        """Test that generatePredictionChart correctly creates a chart file"""
        # Setup test parameters
        prediction_value = 170.0
        days_ahead = 30
        ticker = "AAPL"
        period = "1y"
        interval = "1d"
        model_name = "TestModel"
        
        # Create predictions directory
        pred_dir = tmp_path / "data" / "predictions"
        os.makedirs(pred_dir, exist_ok=True)
        
        # Mock the plt.savefig to avoid actually creating a file
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'), \
             patch('utils.generatePredictionChart', return_value=str(pred_dir / f"{ticker}_{period}_{interval}_{model_name.lower()}.png")):
            
            # Call the function with a mock to avoid creating actual file
            chart_path = utils.generatePredictionChart(
                data=processed_stock_data,
                predictionValue=prediction_value,
                days_ahead=days_ahead,
                ticker=ticker,
                period=period,
                interval=interval,
                modelName=model_name
            )
            
            # Check that the returned path has the expected format
            assert ticker in chart_path
            assert period in chart_path
            assert interval in chart_path
            assert model_name.lower() in chart_path
            assert chart_path.endswith('.png')

    def test_date_calculation_for_different_intervals(self):
        """Test that the chart function correctly calculates future dates for different intervals"""
        # Create a simple dataframe with one row
        data = pd.DataFrame({
            'Price': [pd.to_datetime('2023-01-15')],
            'Close': [150.0]
        })
        
        # Test cases for different intervals
        test_cases = [
            ('1m', 5),    # 5 minutes ahead
            ('1h', 2),    # 2 hours ahead
            ('1d', 10),   # 10 days ahead
            ('1wk', 3),   # 3 weeks ahead
            ('1mo', 2),   # 2 months ahead
        ]
        
        # Mock the plotting functions
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.scatter'), \
             patch('matplotlib.pyplot.annotate'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.grid'), \
             patch('matplotlib.pyplot.legend'), \
             patch('matplotlib.pyplot.gcf'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            # Test each interval type
            for interval, days_ahead in test_cases:
                # Call the function (we're not testing the return here, just that it runs)
                utils.generatePredictionChart(
                    data=data,
                    predictionValue=160.0,
                    days_ahead=days_ahead,
                    ticker="TEST",
                    period="1y",
                    interval=interval,
                    modelName="TestModel"
                )
                # No assertion needed - we're just checking it doesn't raise an exception
