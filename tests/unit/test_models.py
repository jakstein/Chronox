import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import model
import news

class TestXGBoostModel:
    @patch('pandas.read_csv')
    @patch('model.news.getSentimentData')
    @patch('model.news.adjustPredictionWithSentiment')
    @patch('xgboost.XGBRegressor')
    @patch('sklearn.model_selection.train_test_split')
    def test_train_xgboost(self, mock_train_test_split, mock_xgb, mock_adjust, mock_get_sentiment, mock_read_csv, processed_stock_data):
        """Test XGBoost model training and prediction"""
        # Setup mocks
        ticker, period, interval = "AAPL", "1y", "1d"
        day_target, test_size = 30, 0.2
        last_close = 160.0
        prediction = 175.0
        sentiment_score = 0.5
        adjusted_prediction = 178.0
        
        # Mock the pandas read_csv call
        mock_read_csv.return_value = processed_stock_data
        
        # Mock train/test split
        x_train = processed_stock_data.drop(columns=["Price"])
        y_train = pd.Series([150.0] * len(processed_stock_data))
        x_test = x_train.copy()
        y_test = pd.Series([155.0] * len(processed_stock_data))
        mock_train_test_split.return_value = (x_train, x_test, y_train, y_test)
        
        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([prediction])
        mock_xgb.return_value = mock_model
        
        # Mock sentiment data
        mock_get_sentiment.return_value = ({"sentimentScore": sentiment_score, "enabled": True}, [])
        
        # Mock sentiment adjustment
        mock_adjust.return_value = adjusted_prediction
        
        # Call the function with return_result=True
        result = model.trainXGBoost(
            processed_stock_data, ticker, period, interval, day_target, test_size, return_result=True
        )
        
        # Verify result structure
        assert result is not None
        assert result["ticker"] == ticker
        assert result["period"] == period
        assert result["interval"] == interval
        assert result["daysAhead"] == day_target
        assert result["prediction"] == adjusted_prediction
        assert result["originalPrediction"] == prediction
        assert result["sentimentScore"] == sentiment_score
        assert result["sentimentEnabled"] == True
        assert "error" in result

class TestLightGBMModel:
    @patch('pandas.read_csv')
    @patch('model.news.getSentimentData')
    @patch('model.news.adjustPredictionWithSentiment')
    @patch('lightgbm.LGBMRegressor')
    @patch('sklearn.model_selection.train_test_split')
    def test_train_lightgbm(self, mock_train_test_split, mock_lgbm, mock_adjust, mock_get_sentiment, mock_read_csv, processed_stock_data):
        """Test LightGBM model training and prediction"""
        # Setup mocks
        ticker, period, interval = "MSFT", "1y", "1d"
        day_target, test_size = 30, 0.2
        last_close = 280.0
        prediction = 295.0
        sentiment_score = 0.3
        adjusted_prediction = 298.0
        
        # Mock the pandas read_csv call
        mock_read_csv.return_value = processed_stock_data
        
        # Mock train/test split
        x_train = processed_stock_data.drop(columns=["Price"])
        y_train = pd.Series([270.0] * len(processed_stock_data))
        x_test = x_train.copy()
        y_test = pd.Series([275.0] * len(processed_stock_data))
        mock_train_test_split.return_value = (x_train, x_test, y_train, y_test)
        
        # Mock LightGBM model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([prediction])
        mock_lgbm.return_value = mock_model
        
        # Mock sentiment data
        mock_get_sentiment.return_value = ({"sentimentScore": sentiment_score, "enabled": True}, [])
        
        # Mock sentiment adjustment
        mock_adjust.return_value = adjusted_prediction
        
        # Call the function with return_result=True
        result = model.trainLightGBM(
            processed_stock_data, ticker, period, interval, day_target, test_size, return_result=True
        )
        
        # Verify result structure
        assert result is not None
        assert result["ticker"] == ticker
        assert result["period"] == period
        assert result["interval"] == interval
        assert result["daysAhead"] == day_target
        assert result["prediction"] == adjusted_prediction
        assert result["originalPrediction"] == prediction
        assert result["sentimentScore"] == sentiment_score
        assert result["sentimentEnabled"] == True
        assert "error" in result

class TestProphetModel:
    @patch('pandas.read_csv')
    @patch('model.news.getSentimentData')
    @patch('model.news.adjustPredictionWithSentiment')
    @patch('prophet.Prophet')
    def test_train_prophet(self, mock_prophet, mock_adjust, mock_get_sentiment, mock_read_csv, processed_stock_data):
        """Test Prophet model training and prediction"""
        # Setup mocks
        ticker, period, interval = "GOOGL", "1y", "1d"
        day_target, test_size = 30, 0.2
        last_close = 2500.0
        prediction = 2600.0
        sentiment_score = 0.7
        adjusted_prediction = 2650.0
        
        # Mock the pandas read_csv call
        mock_read_csv.return_value = processed_stock_data
        
        # Mock Prophet model and forecast
        mock_model = MagicMock()
        mock_forecast = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=30),
            'yhat': np.linspace(2500, 2600, 30)
        })
        mock_model.predict.return_value = mock_forecast
        mock_model.make_future_dataframe.return_value = pd.DataFrame({'ds': pd.date_range(start='2023-01-01', periods=30)})
        mock_prophet.return_value = mock_model
        
        # Mock sentiment data
        mock_get_sentiment.return_value = ({"sentimentScore": sentiment_score, "enabled": True}, [])
        
        # Mock sentiment adjustment
        mock_adjust.return_value = adjusted_prediction
        
        # Call the function with return_result=True
        with patch('prophet.Prophet.fit'), patch('pandas.Series.iloc', return_value=last_close):
            result = model.trainProphet(
                processed_stock_data, ticker, period, interval, day_target, test_size, return_result=True
            )
        
        # Verify result structure
        assert result is not None
        assert result["ticker"] == ticker
        assert result["period"] == period
        assert result["interval"] == interval
        assert result["daysAhead"] == day_target
        assert "prediction" in result
        assert "originalPrediction" in result
        assert "sentimentScore" in result
        assert "sentimentEnabled" in result
        assert "error" in result
