import pandas, numpy, xgboost, lightgbm
from sklearn import model_selection as sklS
from sklearn import metrics as sklM
import utils
from prophet import Prophet  
import news  


def trainXGBoost(data, ticker, tperiod, tinterval, dayTarget, testSize, return_result=False, shuffle=False):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)
        # get the last closing price
    lastClose = data["Close"].iloc[-1]  

    if dayTarget > 0:
        data["Target"] = data["Close"].shift(-dayTarget)
        data = data.dropna(subset=["Target"]) #remove the empty fields from shifted values
    else:
        data["Target"] = data["Close"]

    y = data["Target"]
    x = data.drop(columns=["Price", "Target"])

    XTrain, XTest, YTrain, YTest = sklS.train_test_split(x, y, test_size=testSize, shuffle=shuffle)

    xgbModel = xgboost.XGBRegressor(objective="reg:squarederror", learning_rate=0.05, n_estimators=200) #adjust estimators
    xgbModel.fit(XTrain, YTrain)
    predXgb = xgbModel.predict(XTest)
    futurefeatures = x.iloc[[-1]]
    futurepred = xgbModel.predict(futurefeatures)[0]
    
    sentimentData, _ = news.getSentimentData(ticker)
    sentimentScore = sentimentData.get('sentimentScore', 0)
    
    adjustedPrediction = news.adjustPredictionWithSentiment(
        prediction=futurepred, 
        sentimentScore=sentimentScore,
        originalPrice=lastClose,
        daysAhead=dayTarget
    )
    
    error = numpy.sqrt(sklM.mean_squared_error(YTest, predXgb))
    if not return_result:
        print(f"Original prediction: {futurepred}")
        print(f"Sentiment-adjusted prediction: {adjustedPrediction}")
        print("Error:")
        print(error)
    
    if return_result:
        return {
            "ticker": ticker,
            "period": tperiod,
            "interval": tinterval,
            "lastClose": lastClose,
            "prediction": adjustedPrediction,  
            "originalPrediction": futurepred,
            "daysAhead": dayTarget,
            "error": error,
            "sentimentScore": sentimentScore,
            "sentimentEnabled": sentimentData.get('enabled', False)
        }


def trainLightGBM(data, ticker, tperiod, tinterval, dayTarget, testSize, seed=42, return_result=False, shuffle=False):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)

    lastClose = data["Close"].iloc[-1] 

    if dayTarget > 0:
        data["Target"] = data["Close"].shift(-dayTarget)
        data = data.dropna(subset=["Target"])
    else:
        data["Target"] = data["Close"]



    y = data["Target"]
    x = data.drop(columns=["Price", "Target"])

    XTrain, XTest, YTrain, YTest = sklS.train_test_split(x, y, test_size=testSize, random_state=seed, shuffle=shuffle)

    lgbModel = lightgbm.LGBMRegressor(objective="regression", learning_rate=0.05, n_estimators=200, random_state=seed)
    lgbModel.fit(XTrain, YTrain)
    predLgb = lgbModel.predict(XTest)
    futurefeatures = x.iloc[[-1]]
    futurepred = lgbModel.predict(futurefeatures)[0]
    
    sentimentData, _ = news.getSentimentData(ticker)
    sentimentScore = sentimentData.get('sentimentScore', 0)
    
    # sentiment adjustment
    adjustedPrediction = news.adjustPredictionWithSentiment(
        prediction=futurepred, 
        sentimentScore=sentimentScore,
        originalPrice=lastClose,
        daysAhead=dayTarget
    )
    
    error = numpy.sqrt(sklM.mean_squared_error(YTest, predLgb))
    if not return_result:
        print(f"Original prediction: {futurepred}")
        print(f"Sentiment-adjusted prediction: {adjustedPrediction}")
        print("Error:")
        print(error)
    
    if return_result:
        return {
            "ticker": ticker,
            "period": tperiod,
            "interval": tinterval,
            "lastClose": lastClose,
            "prediction": adjustedPrediction, 
            "originalPrediction": futurepred,  
            "daysAhead": dayTarget,
            "error": error,
            "sentimentScore": sentimentScore,
            "sentimentEnabled": sentimentData.get('enabled', False)
        }


def trainProphet(data, ticker, tperiod, tinterval, dayTarget, testSize, return_result=False):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', 
                           header=0, 
                           parse_dates=['Price'])
    
    # make a copy of the data and rename columns for Prophet
    prophetData = data[['Price', 'Close']].copy()
    prophetData.columns = ['ds', 'y']
    
    regressorFeatures = ['priceChange', 'ma10', 'ma50', 'ema10', 'ema50', 'macd', 'macdSignal', 
                         'volitStd1w', 'volitStd1mo', 'bollingerUp', 'bollingerDown', 
                         'volma10', 'timeFeature', 'RSI']
      # dodaj regresory
    for feature in regressorFeatures:
        if feature in data.columns:
            prophetData[feature] = data[feature]
    
    # podział na dane treningowe/testowe
    trainSize = int(len(prophetData) * (1 - testSize))
    trainData = prophetData.iloc[:trainSize]
    testData = prophetData.iloc[trainSize:]
    
    # trenuj model Prophet z regresorami
    model = Prophet()
    
    # każda cecha jest regresorem
    for feature in regressorFeatures:
        if feature in prophetData.columns and feature not in ['ds', 'y']:
            model.add_regressor(feature)
    
    model.fit(trainData)
      # wykonaj prognozy dla okresu testowego i przyszłości
    future = model.make_future_dataframe(periods=len(testData) + dayTarget)
    
    # kopiuj wartości regresorów do ramki danych przyszłości dla prognozy
    for feature in regressorFeatures:
        if feature in prophetData.columns and feature not in ['ds', 'y']:
            future[feature] = pandas.Series(prophetData[feature].values)
            # dla horyzontu prognozowania użyj ostatniej wartości dla każdego regresora
            future.loc[len(prophetData):, feature] = prophetData[feature].iloc[-1]
    
    forecast = model.predict(future)
      # obliczanie błędu
    test_predictions = forecast.iloc[trainSize:trainSize+len(testData)]['yhat'].values
    test_actuals = testData['y'].values
    error = numpy.sqrt(sklM.mean_squared_error(test_actuals, test_predictions))
    
    # pobierz prognozę przyszłości
    futurepred = forecast['yhat'].iloc[-1]
    
    sentimentData, _ = news.getSentimentData(ticker)
    sentimentScore = sentimentData.get('sentimentScore', 0)

    lastClose = data["Close"].iloc[-1]
    
    # dostosowanie prognoz na podstawie sentymentu
    adjustedPrediction = news.adjustPredictionWithSentiment(
        prediction=futurepred, 
        sentimentScore=sentimentScore,
        originalPrice=lastClose,
        daysAhead=dayTarget
    )
    
    if not return_result:
        print(f"Original prediction: {futurepred}")
        print(f"Sentiment-adjusted prediction: {adjustedPrediction}")
        print("Error:")
        print(error)
    
    if return_result:
        return {
            "ticker": ticker,
            "period": tperiod,
            "interval": tinterval,
            "lastClose": lastClose,
            "prediction": adjustedPrediction,  
            "originalPrediction": futurepred,  
            "daysAhead": dayTarget,
            "error": error,
            "sentimentScore": sentimentScore,
            "sentimentEnabled": sentimentData.get('enabled', False)
        }
