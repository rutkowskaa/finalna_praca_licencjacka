import Get_Data
import pandas as pd
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class KNN_AR():
    def __init__(self, data:pd.Series=None, params:dict()=None, plot_predicted_insample:bool=False, test_ratio:float=0.7):
        self.params = params
        self.test_ratio = test_ratio
        self.params = params
        self.lags = params["lags"]



        self.all_data = data

        self.data = data[:int(test_ratio*len(self.all_data))]
        self.data_test = data[int(test_ratio*len(self.all_data)):]

        self.X = pd.DataFrame()
        self.X_test = pd.DataFrame()


        self.make_lags(params["lags"])

        print("DŁUGOŚĆ: ", len(self.X_test), len(self.data_test))


    def make_lags(self, lags):
        for i in range(1, lags + 1):
            self.X.insert(loc=len(self.X.columns), column=f"{i}", value=self.data.shift(i))
        self.X = self.X.iloc[lags:]
        self.data = self.data.iloc[lags:]

        self.X_test = self.X.iloc[int(self.test_ratio*len(self.data)):]
        self.X = self.X.iloc[:int(self.test_ratio*len(self.data))]


        self.data = self.data.iloc[:int(self.test_ratio*len(self.data))]
        self.data_test = self.data.iloc[int(self.test_ratio*len(self.data)):]

        #print("AAAAAAAAAAAAAAAAAA", self.data_test, self.X_test)


    def fit(self, params_fit):
        self.model = neighbors.KNeighborsRegressor(n_neighbors=params_fit["k"])
        self.model.fit(X=self.X, y=self.data)
        self.params = params_fit
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna:int, k_max:int=3):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param k_max: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        def RMSE(preds):
            return (1 / int(dlugosc_okna*len(self.data))) * sum(pred ** 2)
        i = 0
        all_preds = np.array([])
        pure_errors = np.array([])
        for k in range(1, k_max):
            pred = np.array([])
            for i in range(0, len(self.data)-int(dlugosc_okna*len(self.data))):
                wyrazenie = i + int(dlugosc_okna*len(self.data))
                if i + int(dlugosc_okna*len(self.data)) > len(self.data):
                    print("BROKEN")
                    break
                train_x = self.X.iloc[i:wyrazenie]
                train_y = self.data.iloc[i:wyrazenie]

                valid = neighbors.KNeighborsRegressor(n_neighbors=k)
                valid.fit(X=train_x, y=train_y)

                lim = self.X.iloc[wyrazenie, :]
                pred = np.append(pred, valid.predict(X=[lim.values]))


            all_preds = np.append(all_preds, [k, pred])
            pure_errors = np.append(pure_errors, [k, RMSE(preds=pred)]) # RMSE
            pure_errors = pure_errors.reshape(-1, 2)

        bledy = np.array(pure_errors[:, 1])
        min_errors = min(bledy)
        optimal_k = np.where(bledy==min_errors)[0][0] + 1
        print("OPTYMALNA WARTOŚĆ PARAMETRU K: ", optimal_k)


        #plt.plot(self.data.values[int(dlugosc_okna*len(self.data)):int(dlugosc_okna*len(self.data))+50])
        #plt.plot(pred[:50], c='r')
        #plt.show()
        print("cross_validation_rolling_window")

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        print("TUTAJ", self.X_test, self.data_test)
        self.raw_forecasts = self.model.predict(self.X_test)
        print("forecast_raw")
        return self.raw_forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")


getter = Get_Data.Get_Data("^IXIC", "2021-01-01", "1d").make_diff()

knn_ar = KNN_AR(data=getter, params={"lags": 15, "k": 1})
knn_ar.cross_validation_rolling_window(dlugosc_okna=1/3, k_max=10)

knn_ar.fit(params_fit={"k": 9})

plt.plot(knn_ar.data_test.values)
plt.plot(knn_ar.forecast_raw(), c='r')
plt.show()

