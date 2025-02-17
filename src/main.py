import ingestion, preprocessing, processing, utils, model



data = ingestion.fetchStock("NVDA", "5y", "1d")
data = preprocessing.cleanData(data[0], data[1], data[2], data[3])
data = processing.addFeatures(data[0], data[1], data[2], data[3])
model.trainXGBoost(data[0], 30, 0.2, 25965)

