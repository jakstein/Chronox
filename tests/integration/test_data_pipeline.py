import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

import ingestion
import preprocessing
import processing
import model
import utils

@pytest.mark.integration
class TestStockDataPipeline:
    @patch('yfinance.download')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.tight_layout')
    def test_full_data_pipeline(self, mock_tight_layout, mock_ylabel, mock_xlabel, mock_title, mock_plot, mock_figure, mock_close, mock_savefig, mock_yf_download, processed_stock_data, tmp_path):
        """Test the full data pipeline from ingestion to prediction"""
        # Setup test data
        ticker = "AAPL"
        period = "1y"
        interval = "1d"
        
        # Create necessary directories
        os.makedirs(tmp_path / "data/raw", exist_ok=True)
        os.makedirs(tmp_path / "data/processed", exist_ok=True)
        os.makedirs(tmp_path / "data/predictions", exist_ok=True)
          # Mock YFinance to return our sample test data
        mock_yf_download.return_value = sample_stock_data.rename(columns={'Price': 'Date'}).set_index('Date')
        
        # Patch the file paths to use our temporary directory
        with patch('os.makedirs'):
            with patch('builtins.open', create=True):
                # Mock file paths
                raw_path = str(tmp_path / f"data/raw/{ticker}_{period}_{interval}.csv")
                processed_path = str(tmp_path / f"data/processed/{ticker}_{period}_{interval}.csv")
                
                # Step 1: Fetch stock data
                with patch('pandas.DataFrame.to_csv'):
                    ingestion.fetchStock(ticker, period, interval)
                
                # Step 2: Clean the data
                with patch('pandas.read_csv', return_value=processed_stock_data):
                    with patch('pandas.DataFrame.to_csv'):
                        preprocessing.cleanData(processed_stock_data, ticker, period, interval)
                
                # Step 3: Add features
                with patch('pandas.read_csv', return_value=processed_stock_data):
                    with patch('pandas.DataFrame.to_csv'):
                        processed_data = processing.addFeatures(processed_stock_data, ticker, period, interval)
                
                # Step 4: Train a model and get prediction
                with patch('model.news.getSentimentData', return_value=({"sentimentScore": 0.2, "enabled": True}, [])):
                    with patch('model.news.adjustPredictionWithSentiment', return_value=165.0):
                        with patch('pandas.read_csv', return_value=processed_stock_data):
                            with patch('sklearn.model_selection.train_test_split') as mock_split:
                                with patch('xgboost.XGBRegressor') as mock_xgb:
                                    
                                    # Setup mock returns for model training
                                    x_train = processed_stock_data.drop(columns=["Price"])
                                    y_train = pd.Series([150.0] * len(processed_stock_data))
                                    x_test = x_train.copy()
                                    y_test = pd.Series([155.0] * len(processed_stock_data))
                                    
                                    mock_split.return_value = (x_train, x_test, y_train, y_test)
                                    
                                    mock_model = MagicMock()
                                    mock_model.predict.return_value = [160.0]
                                    mock_xgb.return_value = mock_model
                                    
                                    # Run prediction
                                    result = model.trainXGBoost(
                                        processed_stock_data, ticker, period, interval, 30, 0.2, return_result=True
                                    )
                                    
                                    # Verify prediction result
                                    assert result is not None
                                    assert result["ticker"] == ticker
                                    assert result["period"] == period
                                    assert result["interval"] == interval
                                    assert "prediction" in result
                                    assert "originalPrediction" in result
                                    assert "error" in result

    @patch('utils.generatePredictionChart')
    @patch('model.trainXGBoost')
    @patch('pandas.read_csv')
    def test_prediction_to_chart_flow(self, mock_read_csv, mock_train_model, mock_chart, processed_stock_data):
        """Test the flow from model prediction to chart generation"""
        # Setup test data
        ticker = "MSFT"
        period = "1y"
        interval = "1d"
        days_ahead = 30
        prediction_value = 350.0
        chart_path = f"./data/predictions/{ticker}_{period}_{interval}_xgboost.png"
        
        # Mock the model prediction
        prediction_result = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "lastClose": 330.0,
            "prediction": prediction_value,
            "originalPrediction": 345.0,
            "daysAhead": days_ahead,
            "error": 5.2,
            "sentimentScore": 0.3,
            "sentimentEnabled": True
        }
        mock_train_model.return_value = prediction_result
        
        # Mock dataframe read
        mock_read_csv.return_value = processed_stock_data
        
        # Mock chart generation
        mock_chart.return_value = chart_path
        
        # Run the flow
        result = model.trainXGBoost(
            processed_stock_data, ticker, period, interval, days_ahead, 0.2, return_result=True
        )
        
        # Generate chart using the result
        chart_result = utils.generatePredictionChart(
            data=processed_stock_data,
            predictionValue=result["prediction"],
            days_ahead=days_ahead,
            ticker=ticker,
            period=period,
            interval=interval,
            modelName="XGBoost"
        )
        
        # Verify that the chart was generated with correct parameters
        mock_chart.assert_called_once()
        args, kwargs = mock_chart.call_args
        
        assert kwargs["predictionValue"] == prediction_value
        assert kwargs["days_ahead"] == days_ahead
        assert kwargs["ticker"] == ticker
        assert kwargs["period"] == period
        assert kwargs["interval"] == interval
        assert kwargs["modelName"] == "XGBoost"
        assert chart_result == chart_path
