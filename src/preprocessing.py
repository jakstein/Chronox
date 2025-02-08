import pandas
import os

def cleanData(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    ticker, tperiod = name_without_ext.split('_')
    data = pandas.read_csv(filepath)
    data = data.dropna(how='all')
    data.to_csv(f'./data/processed/{ticker}_{tperiod}.csv', index=False)
    return data

cleanData(".\\data\\raw\\AAPL_1y.csv")