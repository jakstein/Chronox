# backtrader
import pandas, os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import gc
import shutil
import json

USERS_FILE = 'users.json' # path to the user preferences file

def loadUserPreferences(userId):
    try:
        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f) # create empty file if not exists
            return {}
        with open(USERS_FILE, 'r') as f:
            allPrefs = json.load(f)
            return allPrefs.get(str(userId), {}) # return user's prefs or empty dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {} # return empty dict on error

def saveUserPreferences(userId, preferences):
    try:
        if not os.path.exists(USERS_FILE):
             with open(USERS_FILE, 'w') as f:
                json.dump({}, f) # create file if needed

        with open(USERS_FILE, 'r') as f:
            try:
                allPrefs = json.load(f)
            except json.JSONDecodeError:
                allPrefs = {} # reset if file is corrupted
    except FileNotFoundError:
         allPrefs = {}

    userId_str = str(userId)

    # get existing prefs or start new dict
    userPrefs = allPrefs.get(userId_str, {})
    userPrefs.update(preferences)

    # put updated prefs back
    allPrefs[userId_str] = userPrefs

    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(allPrefs, f, indent=4) # save with formatting
    except IOError as e:
        print(f"Error saving user preferences: {e}")


def getUserPreference(userId, key, default=None):
    prefs = loadUserPreferences(userId)
    return prefs.get(key, default)


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


def cleanupMemory(delete_files=False):
    gc.collect()
    
    # close all matplotlib figures
    plt.close('all')

    if delete_files:
        # reset data directories
        if os.path.exists('./data/raw'):
            shutil.rmtree('./data/raw')
        if os.path.exists('./data/processed'):
            shutil.rmtree('./data/processed')
        if os.path.exists('./data/predictions'):
            shutil.rmtree('./data/predictions')

        # create fresh data directories
        os.makedirs('./data/raw', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        os.makedirs('./data/predictions', exist_ok=True)
        
    # run garbage collection again after file operations
    gc.collect()