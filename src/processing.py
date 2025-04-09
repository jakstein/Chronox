import numpy, pandas
import news

def addFeatures(data, ticker, tperiod, tinterval):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)
    data["priceChange"] = data["Close"].pct_change()*100
    data["ma10"] = data["Close"].rolling(window=10).mean()
    data["ma50"] = data["Close"].rolling(window=50).mean()
    
    data["ema10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["ema50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["ema12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["ema26"] = data["Close"].ewm(span=26, adjust=False).mean()

    data["macd"] = data["ema12"] - data["ema26"]
    data["macdSignal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data = data.drop("ema12", axis=1)
    data = data.drop("ema26", axis=1)

    data["volitStd1w"] = data["Close"].pct_change().rolling(window=7).std()
    data["volitStd1mo"] = data["Close"].pct_change().rolling(window=30).std()

    data["ma20"] = data["Close"].rolling(window=20).mean()
    data["bollingerUp"] = data["ma20"] + (data["Close"].rolling(20).std()*2)
    data["bollingerDown"] = data["ma20"] - (data["Close"].rolling(20).std()*2)
    data = data.drop("ma20", axis=1)

    data["volma10"] = data["Volume"].rolling(10).mean()

    #data["OBV"] = (numpy.sign(data["Close"].diff()) * data["Volume"]).where(data["Close"].diff() != 0).fillna(0).cumsum() #doublecheck, weird results

    data["timeFeature"] = numpy.linspace(0, 1, len(data))

    RSIdelta = data["Close"].diff()
    RSIgain = RSIdelta.where(RSIdelta > 0,0)
    RSIloss = -RSIdelta.where(RSIdelta < 0,0)
    #wilders smoothing
    RSIavgGain = RSIgain.ewm(com=13, adjust=False).mean()
    RSIavgLoss = RSIloss.ewm(com=13, adjust=False).mean()
    RSI = RSIavgGain/RSIavgLoss
    data["RSI"] = 100-(100/(1+RSI))


    data = data.dropna() #ensure no empty values
    data.to_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', index=False, mode='w')
    
    return data

