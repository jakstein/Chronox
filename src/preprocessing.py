import pandas
import os

def cleanData(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    ticker, tperiod, tinterval = name_without_ext.split('_')
    data = pandas.read_csv(filepath)
    data = data.dropna(how='all')
    data = data.drop(data.index[[0, 1]]) #remove 2 useless rows
    data.to_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', index=False)
    return data

cleanData(".\\data\\raw\\AAPL_5y_1d.csv")