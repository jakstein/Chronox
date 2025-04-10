import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import news

class TestNewsFetching:
    @patch('news.get_feed')
    def test_fetch_ticker_news(self, mock_get_feed, sample_news_stories):
        """Test fetching news for a ticker"""
        # Create mock story objects to match what tickertick would return
        mock_stories = []
        for story_dict in sample_news_stories:
            mock_story = MagicMock()
            mock_story.id = story_dict['id']
            mock_story.time = story_dict['time']
            mock_story.url = story_dict['url']
            mock_story.site = story_dict['site']
            mock_story.tags = [story_dict['title']]
            mock_story.description = story_dict['description']
            mock_stories.append(mock_story)
            
        # Set up the mock to return our sample stories
        mock_get_feed.return_value = mock_stories
        
        # Call the function
        ticker = "AAPL"
        days_lookback = 7
        stories_count = 5
        result = news.fetch_ticker_news(ticker, days_lookback, stories_count)
        
        # Verify the query construction and parameters
        mock_get_feed.assert_called_once()
        args, kwargs = mock_get_feed.call_args
        assert kwargs['query'] == f"(and tt:{ticker.lower()} T:curated)"
        assert kwargs['no'] == stories_count
        assert kwargs['hours_ago'] == days_lookback * 24
        
        # Verify the result format
        assert len(result) == len(sample_news_stories)
        for i, story in enumerate(result):
            assert story['id'] == sample_news_stories[i]['id']
            assert story['time'] == sample_news_stories[i]['time']
            assert story['url'] == sample_news_stories[i]['url']
            assert story['site'] == sample_news_stories[i]['site']
            assert story['title'] == sample_news_stories[i]['title']
            assert story['description'] == sample_news_stories[i]['description']
    
    @patch('news.get_feed')
    def test_fetch_ticker_news_error_handling(self, mock_get_feed):
        """Test error handling when fetching news fails"""
        # Make the API call raise an exception
        mock_get_feed.side_effect = Exception("API error")
        
        # Call the function
        result = news.fetch_ticker_news("AAPL", 7, 5)
        
        # Verify the error was handled and an empty list was returned
        assert result == []

class TestSentimentAnalysis:
    def test_analyze_sentiment_with_stories(self, sample_news_stories):
        """Test sentiment analysis with sample stories"""
        # Call the function
        sentiment_data = news.analyze_sentiment(sample_news_stories)
        
        # Verify the result structure
        assert 'compoundAvg' in sentiment_data
        assert 'positiveRatio' in sentiment_data
        assert 'negativeRatio' in sentiment_data
        assert 'neutralRatio' in sentiment_data
        assert 'storyCount' in sentiment_data
        assert 'sentimentScore' in sentiment_data
        
        # Verify the story count
        assert sentiment_data['storyCount'] == len(sample_news_stories)
        
        # Verify the ratios sum to 1
        ratio_sum = sentiment_data['positiveRatio'] + sentiment_data['negativeRatio'] + sentiment_data['neutralRatio']
        assert abs(ratio_sum - 1.0) < 0.001
    
    def test_analyze_sentiment_without_stories(self):
        """Test sentiment analysis with no stories"""
        # Call the function with an empty list
        sentiment_data = news.analyze_sentiment([])
        
        # Verify default values
        assert sentiment_data['compoundAvg'] == 0
        assert sentiment_data['positiveRatio'] == 0
        assert sentiment_data['negativeRatio'] == 0
        assert sentiment_data['neutralRatio'] == 0
        assert sentiment_data['storyCount'] == 0
        assert sentiment_data['sentimentScore'] == 0

class TestPredictionAdjustment:
    @patch('news.loadConfig')
    def test_adjust_prediction_with_sentiment(self, mock_load_config, mock_config):
        """Test adjusting predictions based on sentiment"""
        # Configure the mock config
        mock_load_config.return_value = mock_config
        
        # Test cases
        test_cases = [
            # prediction, sentiment, originalPrice, daysAhead
            (150.0, 0.8, 140.0, 30),  # Positive sentiment, price increase
            (130.0, -0.6, 140.0, 30), # Negative sentiment, price decrease
            (150.0, 0.0, 140.0, 30),  # Neutral sentiment
            (150.0, 0.5, 140.0, 90),  # Longer prediction horizon
        ]
        
        for prediction, sentiment, original_price, days_ahead in test_cases:
            # Call the function
            adjusted = news.adjustPredictionWithSentiment(
                prediction=prediction,
                sentimentScore=sentiment,
                originalPrice=original_price,
                daysAhead=days_ahead
            )
            
            # For positive sentiment and price increase, adjustment should make it higher
            if sentiment > 0 and prediction > original_price:
                assert adjusted >= prediction
            
            # For negative sentiment and price decrease, adjustment should make it lower
            elif sentiment < 0 and prediction < original_price:
                assert adjusted <= prediction
            
            # For neutral sentiment, should be close to original prediction
            elif abs(sentiment) < 0.05:
                assert abs(adjusted - prediction) < 0.01
    
    @patch('news.loadConfig')
    def test_sentiment_disabled(self, mock_load_config, mock_config):
        """Test that when sentiment is disabled, no adjustment occurs"""
        # Modify the config to disable sentiment
        disabled_config = mock_config.copy()
        disabled_config['news_sentiment']['enabled'] = False
        mock_load_config.return_value = disabled_config
        
        # Call the function
        prediction = 150.0
        result = news.adjustPredictionWithSentiment(
            prediction=prediction,
            sentimentScore=0.8,
            originalPrice=140.0,
            daysAhead=30
        )
        
        # Should not adjust the prediction
        assert result == prediction
