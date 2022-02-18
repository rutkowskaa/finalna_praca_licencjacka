import Get_Data
import pandas as pd
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

class KNN_AR():
    def __init__(self, data:pd.Series=None, params:dict()=None, plot_predicted_insample:bool=False):
        self.params = params
        self.data = data
        self.X = pd.DataFrame()
        self.make_lags(params["lags"])
        self.params = params
        self.lags = params["lags"]

    def make_lags(self, lags):
        for i in range(1, lags + 1):
            self.X.insert(loc=len(self.X.columns), column=f"{i}", value=self.data.shift(i))
        self.X = self.X.iloc[lags:, :]
        self.data = self.data.iloc[lags:]

    def fit(self):
        model = neighbors.KNeighborsRegressor(n_neighbors=self.params["k"])
        model.fit(X=self.X, y=self.data)
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna:int, lags:int=None):
        print(len(self.data))
        i = 0
        errors = np.array([])
        for i in range(0, len(self.data)):
            wyrazenie = i + int(dlugosc_okna*len(self.data))
            if i + int(dlugosc_okna*len(self.data)) > len(self.data):
                break
            train_x = self.X.iloc[i:wyrazenie]
            train_y = self.data.iloc[i:wyrazenie]
            valid = neighbors.KNeighborsRegressor(n_neighbors=self.params["k"])
            valid.fit(X=train_x, y=train_y)
            if self.lags == 1:
                errors = np.append(errors, valid.predict(X=train_x.iloc[-1].values.reshape(1,-1)))
        plt.plot(errors)
        plt.show()
        print("cross_validation_rolling_window")

    def predict(self):
        print("predict")

    def forecast_raw(self):
        print("forecast_raw")

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")


getter = Get_Data.Get_Data("^IXIC", "2021-01-01", "1d").make_diff()

knn_ar = KNN_AR(data=getter, params={"lags": 1, "k": 1})
knn_ar.cross_validation_rolling_window(dlugosc_okna=1/3)
