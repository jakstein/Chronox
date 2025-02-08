import yfinance as yf
import matplotlib.pyplot as plt
ticker = "AAPL"
#1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
period = "1y"
def fetchStock(ticker, tperiod):
    tickerData = yf.download(ticker, period=tperiod)
    tickerData.to_csv(f'./data/raw/{ticker}_{tperiod}.csv')
    plt.figure(figsize=(10, 5))
    plt.plot(tickerData.index, tickerData['Close'], label='Close Price')
    plt.plot(tickerData.index, tickerData['Open'], label='Open Price')
    plt.title(f"{ticker} Stock Price - {tperiod}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./data/raw/{ticker}_{tperiod}.png')
    plt.close()


fetchStock(ticker, period)