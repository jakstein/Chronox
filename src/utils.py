# backtrader
import pandas, os


def loadData(filePath):
    filename = os.path.basename(filePath)
    name_without_ext = os.path.splitext(filename)[0]
    ticker, tperiod, tinterval = name_without_ext.split('_')
    data = pandas.read_csv(filePath, header=0, parse_dates=True)
    return data