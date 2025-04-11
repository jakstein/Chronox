from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tickertick import get_feed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import loadConfig

def fetch_ticker_news(ticker: str, daysLookback: int = 7, storiesCount: int = 20) -> List[Dict]:
    try:
        # oblicz godziny wstecz z daysLookback
        hours_ago = daysLookback * 24
        
        # konstrukcja zapytania dla tickertick
        # format: (and tt:<ticker> T:curated)
        query = f"(and tt:{ticker.lower()} T:curated)"
        
        stories = get_feed(query=query, no=storiesCount, hours_ago=hours_ago)
        
        # zwróć historię jako listę słowników
        return [
            {
                'id': story.id,
                'time': story.time,
                'url': story.url,
                'site': story.site,
                'title': story.tags[0] if story.tags else "Brak dostępnego tytułu",
                'description': story.description or "Brak dostępnego opisu"
            }
            for story in stories
        ]
    except Exception as e:
        print(f"Błąd podczas pobierania wiadomości dla {ticker}: {str(e)}")
        return []

def analyze_sentiment(stories: List[Dict]) -> Dict:
    """
    Analizuj sentyment wiadomości używając VADER
    
    Args:
        stories: Lista słowników z wiadomościami
        
    Returns:
        Słownik zawierający metryki sentymentu
    """
    if not stories:
        return {
            'compoundAvg': 0,
            'positiveRatio': 0,
            'negativeRatio': 0,
            'neutralRatio': 0,
            'storyCount': 0,
            'sentimentScore': 0
        }
    
    analyzer = SentimentIntensityAnalyzer()
    
    # oblicz sentyment dla każdej wiadomości
    compoundScores = []
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for story in stories:
        # analizuj zarówno tytuł jak i opis
        title = story.get('title', '')
        description = story.get('description', '')
        content = f"{title}. {description}"
        
        # pobierz wyniki sentymentu
        sentiment = analyzer.polarity_scores(content)
        compoundScores.append(sentiment['compound'])
        
        if sentiment['compound'] >= 0.05:
            sentiments['positive'] += 1
        elif sentiment['compound'] <= -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    
    # oblicz metryki
    storyCount = len(stories)
    compoundAvg = np.mean(compoundScores) if compoundScores else 0
    
    # oblicz współczynniki
    positiveRatio = sentiments['positive'] / storyCount if storyCount > 0 else 0
    negativeRatio = sentiments['negative'] / storyCount if storyCount > 0 else 0
    neutralRatio = sentiments['neutral'] / storyCount if storyCount > 0 else 0
    
    # wynik od -1 do 1
    sentimentScore = compoundAvg
    
    return {
        'compoundAvg': compoundAvg,
        'positiveRatio': positiveRatio,
        'negativeRatio': negativeRatio,
        'neutralRatio': neutralRatio,
        'storyCount': storyCount,
        'sentimentScore': sentimentScore
    }

def getSentimentData(ticker: str) -> Tuple[Dict, List[Dict]]:

    config = loadConfig()
    newsConfig = config.get('news_sentiment', {})
    
    if not newsConfig.get('enabled', False):
        return {
            'compoundAvg': 0,
            'positiveRatio': 0,
            'negativeRatio': 0,
            'neutralRatio': 0,
            'storyCount': 0,
            'sentimentScore': 0,
            'enabled': False
        }, []
        
    daysLookback = newsConfig.get('daysLookback', 7)
    storiesCount = newsConfig.get('storiesCount', 20)
    
    stories = fetch_ticker_news(ticker, daysLookback, storiesCount)
    sentimentData = analyze_sentiment(stories)
    sentimentData['enabled'] = True
    
    return sentimentData, stories

def adjustPredictionWithSentiment(prediction: float, sentimentScore: float, originalPrice: float = None, daysAhead: int = 30, weight: Optional[float] = None) -> float:

    config = loadConfig()
    newsConfig = config.get('news_sentiment', {})
    
    # zwróć oryginalną wartość jeśli sentyment jest wyłączony
    if not newsConfig.get('enabled', False):
        return prediction
    
    weight = weight or newsConfig.get('sentiment_impact_weight', 0.5)
    
    # upewnij się, że waga jest ograniczona
    weight = max(0, min(1, weight))
    
    # zanik wpływu wiadomości
    timeFactor = max(0.2, min(1.0, 30 / max(30, daysAhead)))
    
    adjustedWeight = weight * timeFactor
    
    if originalPrice is not None and originalPrice > 0:
        predicted_change = prediction - originalPrice
        isPriceIncrease = predicted_change > 0
        if isPriceIncrease:
            adjustmentFactor = 1 + (sentimentScore * adjustedWeight)
        else:
            adjustmentFactor = 1 - (sentimentScore * adjustedWeight)
        adjustedChange = predicted_change * adjustmentFactor
        return originalPrice + adjustedChange
    else:
        adjustmentFactor = 1 + (sentimentScore * adjustedWeight / 100)
        return prediction * adjustmentFactor
