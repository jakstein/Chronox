import pandas
import os

def cleanData(data, ticker, tperiod, tinterval):
    data = data.dropna(how="all")
    data = data.drop(data.index[[0, 1]]) #remove 2 useless rows
    data.to_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', index=False)
    return data, ticker, tperiod, tinterval
