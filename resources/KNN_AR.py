import Get_Data
import pandas as pd
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class KNN_AR():
    def __init__(self, data:pd.Series=None, params:dict()=None, plot_predicted_insample:bool=False, test_ratio:float=0.95):
        self.params = params
        self.test_ratio = test_ratio
        self.params = params
        self.lags = params["lags"]


        self.all_data = data.iloc[self.lags:]
        self.all_data_zapas = data

        self.data = data
        self.data1 = data[self.lags:]
        self.data = data[:int(test_ratio*len(self.all_data))]
        self.data_test = self.data1[int(test_ratio*len(self.all_data)):]

        X = self.make_lags(self.all_data_zapas, params["lags"])
        self.all_Xs = X

        self.X = X.iloc[:int(test_ratio*len(self.all_data))]
        self.X_test = X.iloc[int(test_ratio*len(self.all_data)):]


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
        #dlugosc_okna = 1-dlugosc_okna
        self.dlugosc_okna = dlugosc_okna
        def MSE_cross_val(preds):
            actual = self.data[self.prog:]
            mse = (1 / len(preds)) * sum((actual - preds) ** 2)
            return mse
        i = 0
        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = int(dlugosc_okna*len(self.data))
        for k in range(1, k_max):

            pred = np.array([])

            for i in range(int(dlugosc_okna*len(self.data)), len(self.data)):

                train_x = self.X.iloc[:i]
                train_y = self.data.iloc[:i]

                valid = neighbors.KNeighborsRegressor(n_neighbors=k)
                valid.fit(X=train_x, y=train_y)

                lim = self.X.iloc[i, :]
                pred = np.append(pred, valid.predict(X=[lim.values]))


            all_preds = np.append(all_preds, [k, pred])
            pure_errors = np.append(pure_errors, [k, MSE_cross_val(preds=pred)]) # RMSE RMSE_cross_val(preds=pred)]
            pure_errors = pure_errors.reshape(-1, 2)

        bledy = np.array(pure_errors[:, 1])
        min_errors = min(bledy)
        optimal_k = np.where(bledy==min_errors)[0][0] + 1
        #print("Błędy: ", bledy)
        print("OPTYMALNA WARTOŚĆ PARAMETRU K: ", optimal_k)

        print("cross_validation_rolling_window")
        return optimal_k

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])

        for i in range(len(self.data), len(self.all_data)):
            to_test_x = self.all_Xs[i - self.prog : i]
            to_test_y = self.all_data[i - self.prog : i]
            #print(self.data_test)
            #print("------------------------------------------")
            #print("to_test_y")
            #print(to_test_y.iloc[3:])
            #print("------------------------------------------")
            #print("to_test_x")
            #print(to_test_x.iloc[:3])
            #print("------------------------------------------")
            #print("self.datatest")
            #print(self.data_test.iloc[ :len(self.X_test)-3])
            #break
            model = neighbors.KNeighborsRegressor(n_neighbors=self.params['k'])
            model.fit(X=to_test_x, y=to_test_y)
            forecasts = np.append(forecasts, model.predict([self.all_Xs.iloc[i]]))




        print("forecast_raw")
        return forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")


getter = Get_Data.Get_Data("AAPL", "2022-01-01", "1h").make_diff()

knn_ar = KNN_AR(data=getter, params={"lags": 3})
opt = knn_ar.cross_validation_rolling_window(dlugosc_okna=1/3, k_max=15)

knn_ar.fit(params_fit={"k": opt})

plt.plot(knn_ar.data_test.values)
plt.plot(knn_ar.forecast_raw(), c='r')
plt.figure(figsize=(20,10))
plt.show()

