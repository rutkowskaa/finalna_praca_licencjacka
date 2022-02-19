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

        self.data = data
        self.data1 = data[self.lags:]
        self.data = data[:int(test_ratio*len(self.all_data))]
        self.data_test = self.data1[int(test_ratio*len(self.all_data)):]

        X = self.make_lags(self.all_data, params["lags"])

        self.X = X.iloc[:int(test_ratio*len(self.all_data))]
        self.X_test = X.iloc[int(test_ratio*len(self.all_data)):]




        print("DŁUGOŚĆ: ", len(self.X_test), len(self.data_test))


    def make_lags(self, input:pd.DataFrame, lags):
        output = pd.DataFrame()
        for i in range(1, lags + 1):
            output.insert(loc=len(output.columns), column=f"{i}", value=input.shift(i))

        return output[lags:]



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
        dlugosc_okna = 1-dlugosc_okna
        self.dlugosc_okna = 1-dlugosc_okna
        def RMSE_cross_val(preds):
            actual = self.data[self.prog:]
            print(actual, f"prog: {self.prog}")
            print("LEN ", len(preds), len(actual))
            return sum(actual-preds)
        i = 0
        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = len(self.data)-int(dlugosc_okna*len(self.data))
        #print("PROG ", self.data[self.prog:])
        for k in range(1, k_max):

            pred = np.array([])

            for i in range(0, len(self.data)-int(dlugosc_okna*len(self.data))):

                wyrazenie = i + int(dlugosc_okna*len(self.data))
                print("SPRAWDZENIE", wyrazenie)
                if i + int(dlugosc_okna*len(self.data)) > len(self.data):
                    print("BROKEN")
                    break

                train_x = self.X.iloc[i:wyrazenie]
                train_y = self.data.iloc[i:wyrazenie]

                valid = neighbors.KNeighborsRegressor(n_neighbors=k)
                valid.fit(X=train_x, y=train_y)

                lim = self.X.iloc[wyrazenie, :]
                #print("SPRAWDZENIE", lim, train_y.iloc[-1])
                pred = np.append(pred, valid.predict(X=[lim.values]))
                print(i)

            #print("TUTAJ TERAZ ", len(pred))
            all_preds = np.append(all_preds, [k, pred])
            pure_errors = np.append(pure_errors, [k, RMSE_cross_val(preds=pred)]) # RMSE
            pure_errors = pure_errors.reshape(-1, 2)

        bledy = np.array(pure_errors[:, 1])
        min_errors = min(bledy)
        optimal_k = np.where(bledy==min_errors)[0][0] + 1
        print("Błędy: ", bledy)
        print("OPTYMALNA WARTOŚĆ PARAMETRU K: ", optimal_k)

        print("cross_validation_rolling_window")
        return optimal_k

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])

        for i in range(0, len(self.X_test) - int(len(self.X_test)*self.dlugosc_okna)):
            to_test_x = self.X_test[i : i + int(len(self.X_test)*self.dlugosc_okna)]
            to_test_y = self.data_test[i : i + int(len(self.X_test)*self.dlugosc_okna)]
            
            model = neighbors.KNeighborsRegressor(n_neighbors=self.params['k'])
            model.fit(X=to_test_x, y=to_test_y)


        self.raw_forecasts = self.model.predict(self.X_test)
        print("forecast_raw")
        return self.raw_forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")


getter = Get_Data.Get_Data("AAPL", "2022-01-01", "1h").make_diff()

knn_ar = KNN_AR(data=getter, params={"lags": 2})
opt = knn_ar.cross_validation_rolling_window(dlugosc_okna=1/3, k_max=10)

knn_ar.fit(params_fit={"k": opt})

plt.plot(knn_ar.data.values)
plt.plot(knn_ar.predict(), c='r')
plt.show()

plt.plot(knn_ar.data_test.values)
plt.plot(knn_ar.forecast_raw(), c='r')
plt.figure(figsize=(20,10))
plt.show()

