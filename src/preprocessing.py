import pandas
import os

def cleanData(data, ticker, tperiod, tinterval):
    data = pandas.read_csv(f'./data/raw/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)
    data = data.dropna(how="all")
    data = data.drop(data.index[[0, 1]]) # usuń pierwsze dwa wiersze
    data["Close"] = pandas.to_numeric(data["Close"], errors='coerce')
    data.to_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', index=False, mode='w')
