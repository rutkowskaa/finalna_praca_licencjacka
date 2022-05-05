import pandas as pd
import statsmodels.api as sm
import numpy as np
import Get_Data
from statsmodels.tsa.arima.model import ARIMA

class AR_predict():
    def __init__(self, data, p, okno=1/3, test_ratio=0.7):
        train = data[:int(len(data) * test_ratio)]
        test = data[int(len(data) * test_ratio):]
        predictions = np.array([])
        for i in range(int(okno * len(test)), len(test)):
            df = test.iloc[i - int(okno * len(test)): i]
            model = ARIMA(endog=df, order=(p, 0, 0)).fit()
            predictions = np.append(predictions, model.forecast(1))
        self.predictions = predictions
        self.train = train
        self.test = test

    def predict(self):
        return self.predictions