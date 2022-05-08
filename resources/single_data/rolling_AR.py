import pandas as pd
import statsmodels.api as sm
import numpy as np
import Get_Data
from statsmodels.tsa.arima.model import ARIMA

class AR_predict():
    def __init__(self, data, p, okno=1/3, test_ratio=0.7):
        self.forecast_errors = None
        predictions = np.array([])
        self.data = data
        self.p = p
        self.prog = int(okno * len(data))
        self.ratio_int = int(len(data) * test_ratio)
        self.train = data[:self.ratio_int]
        self.test = data[self.ratio_int:]

        for i in range(self.prog, len(self.train)):
            df = data.iloc[i - self.prog : i]
            model = ARIMA(endog=df, order=(p, 0, 0)).fit()
            predictions = np.append(predictions, model.forecast(1))

        self.predictions = predictions

    def predict(self):
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])

        for i in range(len(self.train), len(self.data)):
            to_test = self.data[i - self.prog: i + 2]

            model = ARIMA(endog=to_test, order=(self.p, 0, 0)).fit()
            forecasts = np.append(forecasts, model.forecast(1))

        self.forecast_errors = self.test.values - forecasts

        print("forecast_raw")
        return forecasts

