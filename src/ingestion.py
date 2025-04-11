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
        tickerData.to_csv(f'./data/raw/{ticker}_{tperiod}_{tinterval}.csv', mode='w')
        plt.figure(figsize=(10, 5))
        plt.plot(tickerData.index, tickerData['Close'], label='Cena Zamknięcia')
        plt.plot(tickerData.index, tickerData['Open'], label='Cena Otwarcia')
        plt.title(f"{ticker} Cena Akcji - {tperiod} w interwale {tinterval}")
        plt.xlabel("Data")
        plt.ylabel("Cena (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./data/raw/{ticker}_{tperiod}_{tinterval}.png')
        plt.close()
    else:
        print(f"Nie pobrano danych dla spółki: {ticker}")
