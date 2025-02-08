import yfinance as yf

ticker = "AAPL"
period = "1y"
def fetchStock(ticker, tperiod):
    tickerData = yf.download(ticker, period=tperiod)
    tickerData.to_csv(f'./data/raw/{ticker}_{tperiod}.csv')

fetchStock(ticker, period)