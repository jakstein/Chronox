# backtrader
import pandas, os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


def loadData(filePath):
    filename = os.path.basename(filePath)
    name_without_ext = os.path.splitext(filename)[0]
    ticker, tperiod, tinterval = name_without_ext.split('_')
    data = pandas.read_csv(filePath, header=0, parse_dates=True)
    return data


def generatePredictionChart(data, predictionValue, days_ahead, ticker, period, interval, modelName):
    # oś i figura
    plt.figure(figsize=(12, 6))
    
    # pobierz datę i cenę zamknięcia
    dates = pandas.to_datetime(data['Price'])
    close_prices = data['Close']
    
    # ostatnia data
    lastDate = dates.iloc[-1]
    
    # ustal datę przewidywania
    if interval.endswith('m'):
        minutes = int(interval[:-1])
        futureDate = lastDate + timedelta(minutes=minutes * days_ahead)
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        futureDate = lastDate + timedelta(hours=hours * days_ahead)
    elif interval.endswith('d'):
        futureDate = lastDate + timedelta(days=days_ahead)
    elif interval.endswith('wk'):
        futureDate = lastDate + timedelta(weeks=days_ahead)
    elif interval.endswith('mo'):
        # miesiące po 30 dni
        futureDate = lastDate + timedelta(days=days_ahead * 30)
    else:
        # domyślnie dni, jeśli nieznany
        futureDate = lastDate + timedelta(days=days_ahead)
    
    # historyczne
    plt.plot(dates, close_prices, color='blue', label='Historyczna Cena Zamknięcia')
    
    # linia przyszłości
    plt.plot([lastDate, futureDate], [close_prices.iloc[-1], predictionValue], 
             color='red', linestyle='--', label=f'Przewidywanie {modelName}')
    
    # znacznik punktu predykcji
    plt.scatter(futureDate, predictionValue, color='red', s=50)
    
    # adnotacja przewidywanej wartości
    plt.annotate(f'${predictionValue:.2f}', 
                 (futureDate, predictionValue),
                 xytext=(10, 0),
                 textcoords='offset points',
                 fontweight='bold')
    
    # tytuł i etykiety
    plt.title(f"Przewidywanie Ceny Akcji {ticker} ({modelName})")
    plt.xlabel("Data")
    plt.ylabel("Cena (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # obróć etykiety
    plt.gcf().autofmt_xdate()
    
    # kompaktowy układ
    plt.tight_layout()
    
    chartPath = f'./data/predictions/{ticker}_{period}_{interval}_{modelName.lower()}.png'
    os.makedirs('./data/predictions', exist_ok=True)
    plt.savefig(chartPath)
    plt.close()
    
    return chartPath