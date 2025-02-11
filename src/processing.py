import pandas
import numpy

def addFeatures(filePath):
    data = pandas.read_csv(filePath, header=0, parse_dates=True)

    data["priceChange"] = data["Close"].pct_change()*100
    data["ma10"] = data["Close"].rolling(window=10).mean()
    data["ma50"] = data["Close"].rolling(window=50).mean()
    data["ema10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["ema50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["volitStd1w"] = data["Close"].pct_change().rolling(window=7).std()
    data["volitStd1mo"] = data["Close"].pct_change().rolling(window=30).std()

    data = data.fillna(0)
    data.to_csv(filePath, index=False)

addFeatures(".\\data\\processed\\AAPL_5y_1d.csv")