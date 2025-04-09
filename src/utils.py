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
    # axos and figure
    plt.figure(figsize=(12, 6))
    
    # get date and close price
    dates = pandas.to_datetime(data['Price'])
    close_prices = data['Close']
    
    # last date
    lastDate = dates.iloc[-1]
    
    # figure out prediction date
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
        # months 30 days
        futureDate = lastDate + timedelta(days=days_ahead * 30)
    else:
        # default to days if unknown
        futureDate = lastDate + timedelta(days=days_ahead)
    
    # historical
    plt.plot(dates, close_prices, color='blue', label='Historical Close Price')
    
    # future line
    plt.plot([lastDate, futureDate], [close_prices.iloc[-1], predictionValue], 
             color='red', linestyle='--', label=f'{modelName} Prediction')
    
    # predictiomn point marker
    plt.scatter(futureDate, predictionValue, color='red', s=50)
    
    # annotate predicted value
    plt.annotate(f'${predictionValue:.2f}', 
                 (futureDate, predictionValue),
                 xytext=(10, 0),
                 textcoords='offset points',
                 fontweight='bold')
    
    # title and labels
    plt.title(f"{ticker} Stock Price Prediction ({modelName})")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # rotate lbels
    plt.gcf().autofmt_xdate()
    
    # tight layout
    plt.tight_layout()
    
    chartPath = f'./data/predictions/{ticker}_{period}_{interval}_{modelName.lower()}.png'
    os.makedirs('./data/predictions', exist_ok=True)
    plt.savefig(chartPath)
    plt.close()
    
    return chartPath