import pandas, numpy, xgboost, lightgbm
from sklearn import model_selection as sklS
from sklearn import metrics as sklM
import utils


def trainXGBoost(data, ticker, tperiod, tinterval, dayTarget, testSize, return_result=False):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)    
    if dayTarget > 0:
        data["Target"] = data["Close"].shift(-dayTarget)
        data = data.dropna(subset=["Target"]) #remove the empty fields from shifted values
    else:
        data["Target"] = data["Close"]

    y = data["Target"]
    x = data.drop(columns=["Price", "Target"])

    XTrain, XTest, YTrain, YTest = sklS.train_test_split(x, y, test_size=testSize, shuffle=False)

    xgbModel = xgboost.XGBRegressor(objective="reg:squarederror", learning_rate=0.05, n_estimators=200) #adjust estimators
    xgbModel.fit(XTrain, YTrain)
    predXgb = xgbModel.predict(XTest)
    futurefeatures = x.iloc[[-1]]
    futurepred = xgbModel.predict(futurefeatures)[0]
    
    error = numpy.sqrt(sklM.mean_squared_error(YTest, predXgb))
    if not return_result:
        print(futurepred)
        print("Error:")
        print(error)
    
    if return_result:
        last_close = data["Close"].iloc[-1]
        return {
            "ticker": ticker,
            "period": tperiod,
            "interval": tinterval,
            "last_close": last_close,
            "prediction": futurepred,
            "days_ahead": dayTarget,
            "error": error
        }


def trainLightGBM(data, ticker, tperiod, tinterval, dayTarget, testSize, seed=42, return_result=False):
    data = pandas.read_csv(f'./data/processed/{ticker}_{tperiod}_{tinterval}.csv', header=0, parse_dates=True)
    if dayTarget > 0:
        data["Target"] = data["Close"].shift(-dayTarget)
        data = data.dropna(subset=["Target"])
    else:
        data["Target"] = data["Close"]

    y = data["Target"]
    x = data.drop(columns=["Price", "Target"])

    XTrain, XTest, YTrain, YTest = sklS.train_test_split(x, y, test_size=testSize, random_state=seed, shuffle=False)

    lgbModel = lightgbm.LGBMRegressor(objective="regression", learning_rate=0.05, n_estimators=200, random_state=seed)
    lgbModel.fit(XTrain, YTrain)
    predLgb = lgbModel.predict(XTest)
    futurefeatures = x.iloc[[-1]]
    futurepred = lgbModel.predict(futurefeatures)[0]
    
    error = numpy.sqrt(sklM.mean_squared_error(YTest, predLgb))
    if not return_result:
        print(futurepred)
        print("Error:")
        print(error)
    
    if return_result:
        last_close = data["Close"].iloc[-1]
        return {
            "ticker": ticker,
            "period": tperiod,
            "interval": tinterval,
            "last_close": last_close,
            "prediction": futurepred,
            "days_ahead": dayTarget,
            "error": error
        }
