import yfinance as yf
import matplotlib.pyplot as plt
ticker = "AAPL"
#1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
period = "5y"
interval = "1d"
#1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
def fetchStock(ticker, tperiod, tinterval):
    tickerData = yf.download(ticker, period=tperiod, interval=tinterval)
    if not tickerData.empty:
        tickerData.to_csv(f'./data/raw/{ticker}_{tperiod}_{tinterval}.csv')
        plt.figure(figsize=(10, 5))
        plt.plot(tickerData.index, tickerData['Close'], label='Close Price')
        plt.plot(tickerData.index, tickerData['Open'], label='Open Price')
        plt.title(f"{ticker} Stock Price - {tperiod} at {tinterval} interval")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./data/raw/{ticker}_{tperiod}_{tinterval}.png')
        plt.close()
    else:
        print(f"No data downloaded for ticker: {ticker}")
    return tickerData


fetchStock(ticker, period, interval)