import ingestion, preprocessing, processing, utils, model
ticker = "NVDA"
period = "5y"
interval = "1d"


ingestion.fetchStock(ticker, period, interval)
data = utils.loadData(f'./data/raw/{ticker}_{period}_{interval}.csv')
data = preprocessing.cleanData(data, ticker, period, interval)
data = processing.addFeatures(data, ticker, period, interval)
model.trainXGBoost(data, ticker, period, interval, 30, 0.2, 25965)

